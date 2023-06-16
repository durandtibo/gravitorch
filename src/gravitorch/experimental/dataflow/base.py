from __future__ import annotations

__all__ = ["BaseDataFlow"]

from abc import ABC, abstractmethod
from collections.abc import Iterator
from types import TracebackType


class BaseDataFlow(ABC):
    r"""Base class to implement a dataflow.

    Example usage:

    .. code-block:: pycon

        >>> dataflow: BaseDataFlow = ...
        >>> with dataflow as flow:
        ...     list(flow)
        ...
    """

    def __enter__(self) -> BaseDataFlow:
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
    def __iter__(self) -> Iterator:
        r"""Returns an iterator on the data."""

    @abstractmethod
    def launch(self) -> None:
        r"""Launch the data flow.

        Example usage:

        .. code-block:: pycon

            >>> dataflow: BaseDataFlow = ...
            >>> dataflow.launch()
        """

    @abstractmethod
    def shutdown(self) -> None:
        r"""Shutdown the data flow.

        Example usage:

        .. code-block:: pycon

            >>> dataflow: BaseDataFlow = ...
            >>> dataflow.launch()
            >>> # do anything
            >>> dataflow.shutdown()
        """
