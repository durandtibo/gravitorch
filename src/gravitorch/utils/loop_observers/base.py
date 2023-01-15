__all__ = ["BaseLoopObserver"]

from abc import ABC, abstractmethod
from typing import Any

from objectory import AbstractFactory

from gravitorch.engines.base import BaseEngine


class BaseLoopObserver(ABC, metaclass=AbstractFactory):
    r"""Defines the base class to implement a loop observer.

    The loop observer is designed to work with both training and
    evaluation loops.
    """

    @abstractmethod
    def start(self, engine: BaseEngine) -> None:
        r"""Resets the observer state at the start of each training or
        evaluation loop.

        Args:
            engine (``BaseEngine``): Specifies the engine.
        """

    @abstractmethod
    def end(self, engine: BaseEngine) -> None:
        r"""Performs an action at the end of each training or evaluation loop.

        Args:
            engine (``BaseEngine``): Specifies the engine.
        """

    @abstractmethod
    def update(self, engine: BaseEngine, model_input: Any, model_output: Any) -> None:
        r"""Update the observer.

        Args:
            engine (``BaseEngine``): Specifies the engine.
            model_input: Specifies a batch of model input.
            model_output: Specifies a batch of model output.
        """
