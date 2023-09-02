from __future__ import annotations

__all__ = ["BaseLoopObserver"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from objectory import AbstractFactory

if TYPE_CHECKING:
    from gravitorch.engines import BaseEngine


class BaseLoopObserver(ABC, metaclass=AbstractFactory):
    r"""Defines the base class to implement a loop observer.

    The loop observer is designed to work with both training and
    evaluation loops.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.loops.observers import NoOpLoopObserver
        >>> observer = NoOpLoopObserver()
        >>> observer
        NoOpLoopObserver()
    """

    @abstractmethod
    def start(self, engine: BaseEngine) -> None:
        r"""Resets the observer state at the start of each training or
        evaluation loop.

        Args:
        ----
            engine (``BaseEngine``): Specifies the engine.
        """

    @abstractmethod
    def end(self, engine: BaseEngine) -> None:
        r"""Performs an action at the end of each training or evaluation
        loop.

        Args:
        ----
            engine (``BaseEngine``): Specifies the engine.
        """

    @abstractmethod
    def update(self, engine: BaseEngine, model_input: Any, model_output: Any) -> None:
        r"""Update the observer.

        Args:
        ----
            engine (``BaseEngine``): Specifies the engine.
            model_input: Specifies a batch of model input.
            model_output: Specifies a batch of model output.
        """
