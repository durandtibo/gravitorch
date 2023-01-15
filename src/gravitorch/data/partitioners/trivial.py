__all__ = ["TrivialPartitioner"]

from collections.abc import Sequence
from typing import Optional, TypeVar

from gravitorch.data.partitioners.base import BasePartitioner
from gravitorch.engines import BaseEngine

T = TypeVar("T")


class TrivialPartitioner(BasePartitioner[T]):
    r"""Implements a partitioner that creates the trivial partition."""

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def partition(
        self, items: Sequence[T], engine: Optional[BaseEngine] = None
    ) -> list[Sequence[T]]:
        return [items]
