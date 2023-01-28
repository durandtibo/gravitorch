r"""This module implements PyTorch sampler functions for data loaders."""

__all__ = ["ReproducibleBatchSampler", "PartialSequentialSampler", "PartialRandomSampler"]

from collections.abc import Generator, Iterator

import torch
from torch.utils.data import BatchSampler, Sampler


class ReproducibleBatchSampler(BatchSampler):
    r"""Implements a reproducible batch sampler.

    This class is inspired from PyTorch Ignite. Internally, this class
    iterates and stores indices of the input batch sampler.

    Args:
        batch_sampler (``torch.utils.data.BatchSampler``): batch
            sampler same as used with ``torch.utils.data.DataLoader``
        start_iteration (int, optional): Specifies the starting
            iteration. Default: 0.

    Example usage:

    .. code-block:: python

        >>> from gravitorch.data.dataloaders.samplers import ReproducibleBatchSampler
        >>> from torch.utils.data import BatchSampler, SequentialSampler
        >>> batch_sampler = BatchSampler(
        ...     SequentialSampler(range(10)),
        ...     batch_size=3,
        ...     drop_last=False,
        ... )
        >>> reproducible_batch_sampler = ReproducibleBatchSampler(batch_sampler)
        >>> list(reproducible_batch_sampler)
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> batch_sampler = ReproducibleBatchSampler(batch_sampler, start_iteration=1)
        >>> list(reproducible_batch_sampler)
        [[3, 4, 5], [6, 7, 8], [9]]
        >>> batch_sampler = ReproducibleBatchSampler(batch_sampler, start_iteration=2)
        >>> list(reproducible_batch_sampler)
        [[6, 7, 8], [9]]
    """

    def __init__(self, batch_sampler: BatchSampler, start_iteration: int = 0):
        if not isinstance(batch_sampler, BatchSampler):
            raise TypeError(
                "Argument batch_sampler should be torch.utils.data.sampler.BatchSampler"
            )
        if start_iteration < 0:
            raise ValueError("Argument start_iteration should be positive integer")

        self.batch_sampler = batch_sampler
        self._start_iteration = start_iteration

    def __iter__(self) -> Generator:
        yield from list(self.batch_sampler)[self._start_iteration :]

    def __len__(self) -> int:
        return len(self.batch_sampler)


class PartialSequentialSampler(Sampler):
    r"""Implements a partial sequential sampler that samples only the first
    items in the dataset.

    Args:
        data_source (``torch.utils.data.Dataset``): Specifies the
            dataset to sample from.
        num_samples (int): Specifies the number of samples to draw.
            If the number of samples is bigger than the number of
            samples in the dataset, the number of samples to draw
            is the dataset size.
    """

    def __init__(self, data_source, num_samples: int):
        super().__init__(data_source)
        self.data_source = data_source
        if not isinstance(num_samples, int) or num_samples <= 0:
            raise ValueError(
                f"num_samples should be a positive integer value, but got {num_samples}"
            )
        self.num_samples = num_samples

    def __iter__(self) -> Iterator:
        return iter(range(len(self)))

    def __len__(self) -> int:
        return min(len(self.data_source), self.num_samples)


class PartialRandomSampler(PartialSequentialSampler):
    r"""Implements a partial random sampler that samples randomly some items of
    the dataset.

    Args:
        data_source (``torch.utils.data.Dataset``): Specifies the
            dataset to sample from.
        nnum_samples (int): Specifies the number of samples to draw.
            If the number of samples is bigger than the number of
            samples in the dataset, the number of samples to draw
            is the dataset size.
    """

    def __iter__(self) -> Iterator:
        return iter(torch.randperm(len(self.data_source))[: self.num_samples].tolist())
