__all__ = ["SequentialPartitioner"]

from collections.abc import Sequence
from typing import Optional, TypeVar

from gravitorch.data.partitioners.base import BasePartitioner
from gravitorch.distributed import comm as dist
from gravitorch.engines import BaseEngine

T = TypeVar("T")


class SequentialPartitioner(BasePartitioner[T]):
    r"""Implements a partitioner that select the items in the partition
    sequentially.

    If ``partition_size=1``, the partitioner returns the first item
    for the first epoch, the second items for the second epoch, etc.

    Args:
        partition_size (int, optional): Specifies the partition size
            i.e. the number of items in each partition.
            Default: ``1``
    """

    def __init__(self, partition_size: int = 1):
        self._partition_size = int(partition_size)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(partition_size={self.partition_size:,})"

    @property
    def partition_size(self) -> int:
        r"""int: The partition size."""
        return self._partition_size

    def partition(
        self, items: Sequence[T], engine: Optional[BaseEngine] = None
    ) -> list[Sequence[T]]:
        if not items:
            return [[]]
        world_size = dist.get_world_size()
        epoch = engine.epoch if engine is not None else 0
        num_items = len(items)
        return [
            [
                items[j % num_items]
                for j in range(
                    epoch * world_size * self._partition_size + i * self._partition_size,
                    epoch * world_size * self._partition_size + (i + 1) * self._partition_size,
                )
            ]
            for i in range(world_size)
        ]
