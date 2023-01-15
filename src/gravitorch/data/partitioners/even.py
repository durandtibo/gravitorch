__all__ = ["EvenPartitioner"]

from collections.abc import Sequence
from typing import Optional, TypeVar

from gravitorch.data.partitioners.base import BasePartitioner
from gravitorch.engines import BaseEngine
from gravitorch.utils.partitioning import even_partitions

T = TypeVar("T")


class EvenPartitioner(BasePartitioner[T]):
    r"""Implements a partitioner that creates even partitions i.e. partitions
    with (almost) equal number of items.

    Args:
        num_partitions (int): Specifies the number of partitions.
        drop_remainder (bool, optional): If ``True``, it drops the
            last items if the number of items is not evenly divisible
            by ``num_partitions``.
    """

    def __init__(self, num_partitions: int, drop_remainder: bool = False):
        self._num_partitions = int(num_partitions)
        self._drop_remainder = bool(drop_remainder)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(num_partitions={self._num_partitions:,}, "
            f"drop_remainder={self._drop_remainder})"
        )

    @property
    def drop_remainder(self) -> bool:
        r"""bool: Indicates if the last items are dropped or not if
        the number of items is not evenly divisible by ``num_partitions``.
        """
        return self._drop_remainder

    @property
    def num_partitions(self) -> int:
        r"""int: The number of partitions."""
        return self._num_partitions

    def partition(
        self, items: Sequence[T], engine: Optional[BaseEngine] = None
    ) -> list[Sequence[T]]:
        return even_partitions(
            items=items,
            num_partitions=self._num_partitions,
            drop_remainder=self._drop_remainder,
        )
