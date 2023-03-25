__all__ = ["IterableBatchLoader"]

from collections.abc import Iterable, Iterator
from types import TracebackType
from typing import Optional, TypeVar

from gravitorch.experimental.batchloader.base import BaseBatchLoader

T = TypeVar("T")


class IterableBatchLoader(BaseBatchLoader[T]):
    r"""Implements a simple batch loader that works on any iterable.

    Args:
        iterable (``Iterable``): Specifies the iterable of batches.
    """

    def __init__(self, iterable: Iterable) -> None:
        self._iterable = iterable

    def __enter__(self) -> "IterableBatchLoader":
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        pass

    def __iter__(self) -> Iterator[T]:
        yield from self._iterable

    def __str__(self) -> str:
        return f"{self.__class__.__qualname__}(iterable={self._iterable})"
