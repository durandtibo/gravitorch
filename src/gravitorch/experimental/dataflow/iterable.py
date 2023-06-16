from __future__ import annotations

__all__ = ["IterableDataFlow"]

from collections.abc import Iterable, Iterator

from gravitorch.experimental.dataflow.base import BaseDataFlow


class IterableDataFlow(BaseDataFlow):
    r"""Implements a simple dataflow for iterables.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.experimental.dataflow import IterableDataFlow
        >>> with IterableDataFlow([1, 2, 3, 4, 5]) as dataflow:
        ...     for batch in dataflow:
        ...         print(batch)  # do something
        ...
    """

    def __init__(self, iterable: Iterable) -> None:
        if not isinstance(iterable, Iterable):
            raise TypeError(f"Incorrect type. Expecting iterable but received {type(iterable)}")
        self.iterable = iterable

    def __iter__(self) -> Iterator:
        return iter(self.iterable)

    def __str__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def launch(self) -> None:
        r"""Nothing to do for this dataflow."""

    def shutdown(self) -> None:
        r"""Nothing to do for this dataflow."""
