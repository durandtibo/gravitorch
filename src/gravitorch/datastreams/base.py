from __future__ import annotations

__all__ = ["BaseDataStream"]

from abc import ABC, abstractmethod
from collections.abc import Iterator
from types import TracebackType
from typing import Generic, TypeVar

T = TypeVar("T")


class BaseDataStream(Generic[T], ABC):
    r"""Base class to implement a datastream.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.datastreams import IterableDataStream
        >>> with IterableDataStream([1, 2, 3, 4, 5]) as datastream:
        ...     for batch in datastream:
        ...         print(batch)  # do something
        ...
    """

    def __enter__(self) -> BaseDataStream:
        self.launch()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.shutdown()

    @abstractmethod
    def __iter__(self) -> Iterator[T]:
        r"""Returns an iterator on the data."""

    @abstractmethod
    def launch(self) -> None:
        r"""Launch the datastream.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.datastreams import IterableDataStream
            >>> datastream = IterableDataStream([1, 2, 3, 4, 5])
            >>> datastream.launch()
        """

    @abstractmethod
    def shutdown(self) -> None:
        r"""Shutdown the datastream.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.datastreams import IterableDataStream
            >>> datastream = IterableDataStream([1, 2, 3, 4, 5])
            >>> datastream.launch()
            >>> # do anything
            >>> datastream.shutdown()
        """
