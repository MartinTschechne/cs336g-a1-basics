"""NN utilities."""

from collections.abc import Iterable
from jaxtyping import Float, Int

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F


EPS = 1e-6


def logsumexp(x: Float[Tensor, "..."], dim: int = -1, keepdim: bool = False) -> Float[Tensor, "..."]:
    """
    """
    c = x.max().to(torch.float64)
    return c + torch.log(torch.sum(torch.exp(x.double()-c),dim,keepdim))

def softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    lse = logsumexp(in_features, keepdim=True)
    return torch.exp(in_features.to(torch.float64) - lse)

def cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    batch_size, vocab_size = inputs.shape[0], inputs.shape[-1]
    mask = F.one_hot(targets,vocab_size)
    lse = logsumexp(inputs)
    return -((inputs * mask).sum(-1) - lse).mean()

def get_grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    return np.sqrt(np.sum(np.fromiter(((param.grad.data**2).sum() for param in parameters if param.grad is not None),float)))

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    norm = get_grad_norm(parameters)
    if norm >= max_l2_norm:
        for param in parameters:
            if param.grad is not None:
                param.grad *= max_l2_norm/(norm + EPS)