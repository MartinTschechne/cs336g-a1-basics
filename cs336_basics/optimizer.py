"""Optimizer."""

from typing import Iterable, Optional, Callable, Tuple

import numpy as np
import torch

class AdamW(torch.optim.Optimizer):
    def __init__(self, params: Iterable[torch.nn.Parameter],
                       lr: float=1e-3,
                       weight_decay: float = 0.01,
                       betas: Tuple[float, float] = (0.9,0.999),
                       eps: float = 1e-8):
        super().__init__(params, {
            "lr": lr,
            "beta_1": betas[0],
            "beta_2": betas[1],
            "weight_decay": weight_decay,
            "eps": eps
        })
        if lr < 0:
            raise ValueError("Invalid learning rate {lr}")

    def step(self, closure: Optional[Callable] = None):
        for group in self.param_groups:
            lr = group["lr"]
            beta_1 = group["beta_1"]
            beta_2 = group["beta_2"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                grad = p.grad.data

                m = state.get("m",torch.zeros_like(grad))
                m = beta_1 * m + (1-beta_1) * grad
                state["m"] = m

                v = state.get("v", torch.zeros_like(grad))
                v = beta_2 * v + (1-beta_2) * grad.pow(2)
                state["v"] = v

                t = state.get("t",1)
                alpha_t = lr * (np.sqrt(1-beta_2**t))/(1-beta_1**t)

                p.data -= alpha_t * m/(torch.sqrt(v)+group["eps"])
                p.data -= lr * weight_decay * p.data

                state["t"] = state.get("t",1) + 1

def get_lr_cosine_schedule(it: int,
        max_learning_rate: float,
        min_learning_rate: float,
        warmup_iters: int,
        cosine_cycle_iters: int,
    ) -> float:
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    if it < warmup_iters:
        return it * max_learning_rate / warmup_iters
    elif warmup_iters <= it <= cosine_cycle_iters:
        return min_learning_rate + 0.5*(1+np.cos((it-warmup_iters)*np.pi/(cosine_cycle_iters-warmup_iters)))*(max_learning_rate-min_learning_rate)
    else:
        return min_learning_rate