from __future__ import annotations

__all__ = ["SequentialLoopObserver"]

from collections.abc import Sequence
from typing import Any

from coola.utils import str_indent, str_sequence

from gravitorch.engines.base import BaseEngine
from gravitorch.loops.observers.base import BaseLoopObserver
from gravitorch.loops.observers.factory import setup_loop_observer


class SequentialLoopObserver(BaseLoopObserver):
    r"""Implements a loop observer that is used to run a sequence of loop
    observers.

    This loop observer is designed to run multiple loop observers.

    Args:
    ----
        observers (sequence): Specifies the loop observers or their
            configurations.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.loops.observers import NoOpLoopObserver
        >>> observer = SequentialLoopObserver([NoOpLoopObserver()])
        >>> observer
        SequentialLoopObserver(
          (0): NoOpLoopObserver()
        )
    """

    def __init__(self, observers: Sequence[BaseLoopObserver | dict]) -> None:
        self._observers: tuple[BaseLoopObserver, ...] = tuple(
            setup_loop_observer(observer) for observer in observers
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(\n  {str_indent(str_sequence(self._observers))}\n)"

    def start(self, engine: BaseEngine) -> None:
        for observer in self._observers:
            observer.start(engine)

    def end(self, engine: BaseEngine) -> None:
        for observer in self._observers:
            observer.end(engine)

    def update(self, engine: BaseEngine, model_input: Any, model_output: Any) -> None:
        for observer in self._observers:
            observer.update(engine, model_input, model_output)
