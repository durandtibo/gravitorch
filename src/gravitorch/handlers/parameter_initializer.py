r"""This module implements a handler to initialize parameters."""

__all__ = ["ParameterInitializer"]

import logging
from typing import Union

from gravitorch.engines.base import BaseEngine
from gravitorch.engines.events import EngineEvents
from gravitorch.handlers.base import BaseHandler
from gravitorch.handlers.utils import add_unique_event_handler
from gravitorch.utils.events import VanillaEventHandler
from gravitorch.utils.format import str_indent
from gravitorch.utils.parameter_initializers import (
    BaseParameterInitializer,
    setup_parameter_initializer,
)

logger = logging.getLogger(__name__)


class ParameterInitializer(BaseHandler):
    r"""Implements a handler to initialize the parameters.

    This handler uses a ``BaseParameterInitializer`` object to
    initialize parameters.

    Args:
        parameter_initializer (``BaseParameterInitializer`` or dict):
            Specifies the parameter initializer or its configuration.
        event (str, optional): Specifies the event when to initialize
            the parameters. Default: ``'train_started'``
    """

    def __init__(
        self,
        parameter_initializer: Union[BaseParameterInitializer, dict],
        event: str = EngineEvents.TRAIN_STARTED,
    ):
        self._parameter_initializer = setup_parameter_initializer(parameter_initializer)
        self._event = event

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  parameter_initializer={str_indent(self._parameter_initializer)},"
            f"  event={self._event},"
            ")"
        )

    def attach(self, engine: BaseEngine) -> None:
        add_unique_event_handler(
            engine=engine,
            event=self._event,
            event_handler=VanillaEventHandler(
                self._parameter_initializer.initialize,
                handler_kwargs={"engine": engine},
            ),
        )
