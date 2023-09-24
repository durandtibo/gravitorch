r"""This module implements PyTorch sampler functions for data
loaders."""

from __future__ import annotations

__all__ = ["ReproducibleBatchSampler", "PartialSequentialSampler", "PartialRandomSampler"]

from collections.abc import Generator, Iterator, Sized

import torch
from coola.utils import str_indent, str_mapping
from torch.utils.data import BatchSampler, Sampler


class ReproducibleBatchSampler(BatchSampler):
    r"""Implements a reproducible batch sampler.

    This class is inspired from PyTorch Ignite. Internally, this class
    iterates and stores indices of the input batch sampler.

    Args:
    ----
        batch_sampler (``torch.utils.data.BatchSampler``): batch
            sampler same as used with ``torch.utils.data.DataLoader``
        start_iteration (int, optional): Specifies the starting
            iteration. Default: 0.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.dataloaders.samplers import ReproducibleBatchSampler
        >>> from torch.utils.data import BatchSampler, SequentialSampler
        >>> batch_sampler = BatchSampler(
        ...     SequentialSampler(range(10)),
        ...     batch_size=3,
        ...     drop_last=False,
        ... )
        >>> reproducible_batch_sampler = ReproducibleBatchSampler(batch_sampler)
        >>> reproducible_batch_sampler  # doctest:+ELLIPSIS
        ReproducibleBatchSampler(
          (batch_sampler): <torch.utils.data.sampler.BatchSampler object at 0x...>
          (start_iteration): 0
        )
        >>> list(reproducible_batch_sampler)
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> reproducible_batch_sampler = ReproducibleBatchSampler(batch_sampler, start_iteration=1)
        >>> list(reproducible_batch_sampler)
        [[3, 4, 5], [6, 7, 8], [9]]
        >>> reproducible_batch_sampler = ReproducibleBatchSampler(batch_sampler, start_iteration=2)
        >>> list(reproducible_batch_sampler)
        [[6, 7, 8], [9]]
    """

    def __init__(self, batch_sampler: BatchSampler, start_iteration: int = 0) -> None:
        if not isinstance(batch_sampler, BatchSampler):
            raise TypeError(
                "Argument batch_sampler should be torch.utils.data.sampler.BatchSampler"
            )
        if start_iteration < 0:
            raise ValueError("Argument start_iteration should be positive integer")

        self.batch_sampler = batch_sampler
        self._start_iteration = start_iteration

    def __repr__(self) -> str:
        args = str_indent(
            str_mapping(
                {"batch_sampler": self.batch_sampler, "start_iteration": self._start_iteration}
            )
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def __iter__(self) -> Generator:
        yield from list(self.batch_sampler)[self._start_iteration :]

    def __len__(self) -> int:
        return len(self.batch_sampler)


class PartialSequentialSampler(Sampler):
    r"""Implements a partial sequential sampler that samples only the
    first items in the dataset.

    Args:
    ----
        datasource (``Sized``): Specifies the dataset to sample
            from.
        num_samples (int): Specifies the number of samples to draw.
            If the number of samples is bigger than the number of
            samples in the dataset, the number of samples to draw
            is the dataset size.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.dataloaders.samplers import PartialSequentialSampler
        >>> sampler = PartialSequentialSampler(list(range(10)), num_samples=5)
        >>> sampler
        PartialSequentialSampler(num_samples=5)
        >>> list(sampler)
        [0, 1, 2, 3, 4]
    """

    def __init__(self, datasource: Sized, num_samples: int) -> None:
        super().__init__(datasource)
        self.datasource = datasource
        if not isinstance(num_samples, int) or num_samples <= 0:
            raise ValueError(
                f"num_samples should be a positive integer value, but got {num_samples}"
            )
        self.num_samples = num_samples

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(num_samples={self.num_samples:,})"

    def __iter__(self) -> Iterator:
        return iter(range(len(self)))

    def __len__(self) -> int:
        return min(len(self.datasource), self.num_samples)


class PartialRandomSampler(PartialSequentialSampler):
    r"""Implements a partial random sampler that samples randomly some
    items of the dataset.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.dataloaders.samplers import PartialSequentialSampler
        >>> sampler = PartialSequentialSampler(list(range(10)), num_samples=5)
        >>> sampler
        PartialSequentialSampler(num_samples=5)
        >>> len(list(sampler))
        5
    """

    def __iter__(self) -> Iterator:
        return iter(torch.randperm(len(self.datasource))[: self.num_samples].tolist())

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(num_samples={self.num_samples:,})"
