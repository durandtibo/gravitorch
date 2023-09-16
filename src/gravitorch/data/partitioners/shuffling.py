from __future__ import annotations

__all__ = ["EpochShufflePartitioner"]

from collections.abc import Sequence
from typing import TypeVar

import torch
from coola.utils import str_indent, str_mapping

from gravitorch.data.partitioners.base import BasePartitioner, setup_partitioner
from gravitorch.engines import BaseEngine
from gravitorch.utils.seed import get_torch_generator

T = TypeVar("T")


class EpochShufflePartitioner(BasePartitioner[T]):
    r"""Implements a partitioner that shuffles the data before to
    partition them.

    To be reproducible, the shuffling is controlled by the engine
    epoch value and a base random seed. If no engine is provided,
    the epoch value is set to ``0``.

    Args:
    ----
        partitioner (``BasePartitioner`` or dict): Specifies a
            partitioner or its configuration.
        random_seed (int, optional): Specifies the base random seed.
            This random seed is added to the engine epoch value to
            define the initial seed of the ``torch.Generator`` object.
            Default: ``7553907118525846636``

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.data.partitioners import EpochShufflePartitioner, FixedSizePartitioner
        >>> partitioner = EpochShufflePartitioner(partitioner=FixedSizePartitioner(3))
        >>> partitioner
        EpochShufflePartitioner(
          (partitioner): FixedSizePartitioner(partition_size=3, drop_last=False)
          (random_seed): 7553907118525846636
        )
        >>> partitions = partitioner.partition(list(range(10)))
        >>> partitions
        [[...], [...], [...], [...]]
    """

    def __init__(
        self, partitioner: BasePartitioner | dict, random_seed: int = 7553907118525846636
    ) -> None:
        self._partitioner = setup_partitioner(partitioner)
        self._random_seed = random_seed

    def __repr__(self) -> str:
        args = str_indent(
            str_mapping({"partitioner": self._partitioner, "random_seed": self._random_seed})
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    @property
    def partitioner(self) -> BasePartitioner:
        r"""``BasePartitioner``: The partitioner."""
        return self._partitioner

    @property
    def random_seed(self) -> int:
        r"""``int``: The base random seed."""
        return self._random_seed

    def partition(self, items: Sequence[T], engine: BaseEngine | None = None) -> list[Sequence[T]]:
        epoch = engine.epoch if engine is not None else 0
        permutation = torch.randperm(
            len(items), generator=get_torch_generator(self._random_seed + epoch)
        )
        return self._partitioner.partition([items[i] for i in permutation], engine)
