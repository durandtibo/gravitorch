from __future__ import annotations

__all__ = ["TrivialPartitioner"]

from collections.abc import Sequence
from typing import TypeVar

from gravitorch.data.partitioners.base import BasePartitioner
from gravitorch.engines import BaseEngine

T = TypeVar("T")


class TrivialPartitioner(BasePartitioner[T]):
    r"""Implements a partitioner that creates the trivial partition.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.data.partitioners import TrivialPartitioner
        >>> partitioner = TrivialPartitioner()
        >>> partitioner
        TrivialPartitioner()
        >>> partitions = partitioner.partition(list(range(10)))
        >>> partitions
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def partition(self, items: Sequence[T], engine: BaseEngine | None = None) -> list[Sequence[T]]:
        return [items]
