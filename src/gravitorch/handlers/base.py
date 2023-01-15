__all__ = ["BaseHandler"]

from abc import ABC, abstractmethod

from objectory import AbstractFactory

from gravitorch.engines.base import BaseEngine


class BaseHandler(ABC, metaclass=AbstractFactory):
    r"""Define the base class for the handlers."""

    @abstractmethod
    def attach(self, engine: BaseEngine) -> None:
        r"""Attaches the handler to the engine.

        Args:
            engine (``BaseEngine``): Specifies the engine used to
                attach the handler.
        """
