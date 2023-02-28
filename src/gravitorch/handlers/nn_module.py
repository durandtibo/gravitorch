__all__ = ["ModelModuleFreezer"]

import logging

from gravitorch.engines.base import BaseEngine
from gravitorch.engines.events import EngineEvents
from gravitorch.handlers.base import BaseHandler
from gravitorch.handlers.utils import add_unique_event_handler
from gravitorch.nn import freeze_module
from gravitorch.utils.events import VanillaEventHandler

logger = logging.getLogger(__name__)


class ModelModuleFreezer(BaseHandler):
    r"""Implements a handler to freeze a submodule of the model.

    Args:
        module_name (str): Specifies the name of the module to freeze.
        event (str, optional): Specifies the event when the module
            is frozen. Default: ``'train_started'``
    """

    def __init__(self, module_name: str, event: str = EngineEvents.TRAIN_STARTED):
        self._module_name = str(module_name)
        self._event = str(event)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(event={self._event}, module_name={self._module_name})"
        )

    def attach(self, engine: BaseEngine) -> None:
        add_unique_event_handler(
            engine=engine,
            event=self._event,
            event_handler=VanillaEventHandler(
                self.freeze,
                handler_kwargs={"engine": engine},
            ),
        )

    def freeze(self, engine: BaseEngine) -> None:
        r"""Freezes the module."""
        freeze_module(engine.model.get_submodule(self._module_name))
