"""Common utilities."""

import os
import argparse
import functools
import json
import logging
import time
from collections.abc import Callable
from typing import IO, Any, BinaryIO, Optional

import torch


def stopwatch(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Measure the execution time of any function"""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs) -> Callable[..., Any]:
        start_time = time.time()
        result = fn(*args, **kwargs)
        end_time = time.time()
        logging.info(
            "Function %s took %.3f seconds to execute",
            fn.__name__,
            end_time - start_time,
        )
        return result

    return wrapper


def save_argparse(args: argparse.Namespace, out_path: str) -> None:
    """Serializes the argparse.Namespace to a JSON file.

    Args:
        args: The parsed command-line arguments.
        out_path: The path to save the JSON file.
    """
    config_dict = vars(args)
    with open(out_path, "w") as f:
        json.dump(config_dict, f)


def get_device() -> torch.device:
    device_str = "cpu"
    if torch.cuda.is_available():
        device_str = "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device_str = "mps"
    return torch.device(device_str)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    torch.save({
            'epoch': iteration,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    checkpoint = torch.load(src, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']
