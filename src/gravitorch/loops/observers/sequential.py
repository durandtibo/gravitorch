__all__ = ["SequentialLoopObserver"]

from collections.abc import Sequence
from typing import Any, Union

from gravitorch.engines.base import BaseEngine
from gravitorch.loops.observers.base import BaseLoopObserver
from gravitorch.loops.observers.factory import setup_loop_observer
from gravitorch.utils.format import str_indent, str_torch_sequence


class SequentialLoopObserver(BaseLoopObserver):
    r"""Implements a loop observer that is used to run a sequence of loop
    observers.

    This loop observer is designed to run multiple loop observers.

    Args:
    ----
        observers (sequence): Specifies the loop observers or their
            configurations.
    """

    def __init__(self, observers: Sequence[Union[BaseLoopObserver, dict]]) -> None:
        self._observers: tuple[BaseLoopObserver, ...] = tuple(
            setup_loop_observer(observer) for observer in observers
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  {str_indent(str_torch_sequence(self._observers))}\n)"
        )

    def start(self, engine: BaseEngine) -> None:
        for observer in self._observers:
            observer.start(engine)

    def end(self, engine: BaseEngine) -> None:
        for observer in self._observers:
            observer.end(engine)

    def update(self, engine: BaseEngine, model_input: Any, model_output: Any) -> None:
        for observer in self._observers:
            observer.update(engine, model_input, model_output)
