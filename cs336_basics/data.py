"""Data processing."""

import pathlib
import random
from enum import Enum
from itertools import batched

import numpy as np
import numpy.typing as npt
import torch


class LearningTask(Enum):
    TINY_STORIES_V2 = "TinyStoriesV2-GPT4"
    OWT = "owt"

DATA_PATH = pathlib.Path(__file__).resolve().parent.parent / "data"
FIXTURES_PATH = pathlib.Path(__file__).resolve().parent.parent / "tests/fixtures"

def get_paths(task: LearningTask, debug: bool):
    if debug:
        train_path = FIXTURES_PATH / f"corpus.en"
        valid_path = train_path
        vocab_path = DATA_PATH / "corpus-vocab.json"
        merges_path = DATA_PATH / "corpus-merges.txt"
    else:
        train_path = DATA_PATH / f"{task.value}-train.txt"
        valid_path = DATA_PATH / f"{task.value}-valid.txt"
        vocab_path = DATA_PATH / f"{task.value}-vocab.json"
        merges_path = DATA_PATH / f"{task.value}-merges.txt"
    return train_path, valid_path, vocab_path, merges_path


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    input_seq, labels = [], []
    cl = context_length
    for start in random.sample(range(len(dataset)-cl),k=batch_size):
        input_seq.append(dataset[start:start+cl])
        labels.append(dataset[start+1:start+cl+1])
    return (torch.LongTensor(np.array(input_seq)).to(device), torch.LongTensor(np.array(labels)).to(device))

def linear_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
    input_seq, labels = [], []
    cl = context_length
    for batch in batched(range(0,len(dataset)-cl,cl),batch_size):
        input_seq, labels = [], []
        for start in batch:
            input_seq.append(dataset[start:start+cl])
            labels.append(dataset[start+1:start+cl+1])
        yield (torch.LongTensor(np.array(input_seq)).to(device), torch.LongTensor(np.array(labels)).to(device))