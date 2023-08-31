r"""This module implements a handler to initialize model's
parameters."""

__all__ = ["ModelInitializer"]

import logging
from typing import Union

from coola.utils import str_indent
from minevent import EventHandler

from gravitorch.engines.base import BaseEngine
from gravitorch.engines.events import EngineEvents
from gravitorch.handlers.base import BaseHandler
from gravitorch.handlers.utils import add_unique_event_handler
from gravitorch.nn.init import BaseInitializer, setup_initializer

logger = logging.getLogger(__name__)


class ModelInitializer(BaseHandler):
    r"""Implements a handler to initialize the model's parameters.

    This handler uses a ``BaseInitializer`` object to
    initialize model's parameters.

    Args:
    ----
        initializer (``BaseInitializer`` or dict): Specifies the
            model's parameters initializer or its configuration.
        event (str, optional): Specifies the event when to initialize
            the model's parameters. Default: ``'train_started'``
    """

    def __init__(
        self,
        initializer: Union[BaseInitializer, dict],
        event: str = EngineEvents.TRAIN_STARTED,
    ) -> None:
        self._initializer = setup_initializer(initializer)
        self._event = event

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  initializer={str_indent(self._initializer)},"
            f"  event={self._event},"
            ")"
        )

    def attach(self, engine: BaseEngine) -> None:
        add_unique_event_handler(
            engine=engine,
            event=self._event,
            event_handler=EventHandler(
                self._initializer.initialize,
                handler_kwargs={"module": engine.model},
            ),
        )
