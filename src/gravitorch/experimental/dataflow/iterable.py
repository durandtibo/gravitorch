from __future__ import annotations

__all__ = ["IterableDataFlow"]

from collections.abc import Iterable, Iterator

from gravitorch.experimental.dataflow.base import BaseDataFlow


class IterableDataFlow(BaseDataFlow):
    r"""Implements a simple dataflow for iterables."""

    def __init__(self, iterable: Iterable) -> None:
        if not isinstance(iterable, Iterable):
            raise TypeError(f"{iterable} is not an iterable")
        self.iterable = iterable

    def __iter__(self) -> Iterator:
        return iter(self.iterable)

    def __str__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def launch(self) -> None:
        r"""Nothing to do for this data flow."""

    def shutdown(self) -> None:
        r"""Nothing to do for this data flow."""
