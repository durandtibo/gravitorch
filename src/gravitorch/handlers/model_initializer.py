r"""This module implements a handler to initialize model's
parameters."""

from __future__ import annotations

__all__ = ["ModelInitializer"]

import logging
from typing import TYPE_CHECKING

from coola.utils import str_indent, str_mapping

from gravitorch.engines.events import EngineEvents
from gravitorch.handlers.base import BaseHandler
from gravitorch.handlers.utils import add_unique_event_handler
from gravitorch.nn.init import BaseInitializer, setup_initializer
from gravitorch.utils.events import GEventHandler

if TYPE_CHECKING:
    from gravitorch.engines import BaseEngine

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

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.handlers import ModelInitializer
        >>> from gravitorch.nn.init import Constant
        >>> from gravitorch.testing import create_dummy_engine
        >>> engine = create_dummy_engine()
        >>> handler = ModelInitializer(Constant())
        >>> handler
        ModelInitializer(
          (initializer): Constant(value=0.0, learnable_only=True, log_info=False)
          (event): train_started
        )
        >>> handler.attach(engine)
        >>> engine.trigger_event("train_started")
        >>> engine.model.state_dict()
        OrderedDict([('linear.weight', tensor([[0., 0., 0., 0.],
                [0., 0., 0., 0.],
                [0., 0., 0., 0.]])), ('linear.bias', tensor([0., 0., 0.]))])
    """

    def __init__(
        self,
        initializer: BaseInitializer | dict,
        event: str = EngineEvents.TRAIN_STARTED,
    ) -> None:
        self._initializer = setup_initializer(initializer)
        self._event = event

    def __repr__(self) -> str:
        args = str_indent(
            str_mapping(
                {
                    "initializer": self._initializer,
                    "event": self._event,
                }
            )
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def attach(self, engine: BaseEngine) -> None:
        add_unique_event_handler(
            engine=engine,
            event=self._event,
            event_handler=GEventHandler(
                self._initializer.initialize,
                handler_kwargs={"module": engine.model},
            ),
        )
