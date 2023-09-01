from __future__ import annotations

__all__ = ["BaseHandler"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from objectory import AbstractFactory

if TYPE_CHECKING:
    from gravitorch.engines import BaseEngine


class BaseHandler(ABC, metaclass=AbstractFactory):
    r"""Defines the base class for the handlers.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.handlers import EpochLRMonitor
        >>> from gravitorch.testing import create_dummy_engine
        >>> engine = create_dummy_engine()
        >>> handler = EpochLRMonitor()
        >>> handler
        EpochLRMonitor(freq=1, event=train_epoch_started)
        >>> handler.attach(engine)
    """

    @abstractmethod
    def attach(self, engine: BaseEngine) -> None:
        r"""Attaches the handler to the engine.

        Args:
        ----
            engine (``BaseEngine``): Specifies the engine used to
                attach the handler.
        """
