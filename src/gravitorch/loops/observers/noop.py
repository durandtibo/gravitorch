from __future__ import annotations

__all__ = ["NoOpLoopObserver"]

from typing import Any

from gravitorch.engines.base import BaseEngine
from gravitorch.loops.observers.base import BaseLoopObserver


class NoOpLoopObserver(BaseLoopObserver):
    r"""Implements a no-operation loop observer.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.loops.observers import NoOpLoopObserver
        >>> observer = NoOpLoopObserver()
        >>> observer
        NoOpLoopObserver()
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def start(self, engine: BaseEngine) -> None:
        r"""It is a no-operation method."""

    def end(self, engine: BaseEngine) -> None:
        r"""It is a no-operation method."""

    def update(self, engine: BaseEngine, model_input: Any, model_output: Any) -> None:
        r"""It is a no-operation method."""
