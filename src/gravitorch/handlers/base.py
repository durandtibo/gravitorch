from __future__ import annotations

__all__ = ["BaseHandler"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from objectory import AbstractFactory

if TYPE_CHECKING:
    from gravitorch.engines import BaseEngine


class BaseHandler(ABC, metaclass=AbstractFactory):
    r"""Define the base class for the handlers."""

    @abstractmethod
    def attach(self, engine: BaseEngine) -> None:
        r"""Attaches the handler to the engine.

        Args:
        ----
            engine (``BaseEngine``): Specifies the engine used to
                attach the handler.
        """
