from __future__ import annotations

__all__ = ["ModelFreezer"]

import logging
from typing import TYPE_CHECKING

from gravitorch.engines.events import EngineEvents
from gravitorch.handlers.base import BaseHandler
from gravitorch.handlers.utils import add_unique_event_handler
from gravitorch.nn import freeze_module
from gravitorch.utils.events import GEventHandler

if TYPE_CHECKING:
    from gravitorch.engines import BaseEngine

logger = logging.getLogger(__name__)


class ModelFreezer(BaseHandler):
    r"""Implements a handler to freeze the model or one of its
    submodules.

    Args:
    ----
        event (str, optional): Specifies the event when the model
            is frozen. Default: ``'train_started'``
        module_name (str, optional): Specifies the name of the module
            to freeze if only one of the submodules should be frozen.
            Default: ``''``

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.handlers import ModelFreezer
        >>> from gravitorch.testing import create_dummy_engine
        >>> engine = create_dummy_engine()
        >>> engine.model
        DummyClassificationModel(
          (linear): Linear(in_features=4, out_features=3, bias=True)
          (criterion): CrossEntropyLoss()
        )
        >>> handler = ModelFreezer()
        >>> handler
        ModelFreezer(event=train_started, module_name=)
        >>> handler.attach(engine)
        >>> engine.trigger_event("train_started")
        >>> for name, param in engine.model.named_parameters():
        ...     print(name, param.requires_grad)
        ...
        linear.weight False
        linear.bias False
    """

    def __init__(
        self,
        event: str = EngineEvents.TRAIN_STARTED,
        module_name: str = "",
    ) -> None:
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
            event_handler=GEventHandler(
                self.freeze,
                handler_kwargs={"engine": engine},
            ),
        )

    def freeze(self, engine: BaseEngine) -> None:
        r"""Freezes the model or one of its submodules.

        Args:
        ----
            engine (``BaseEngine``): Specifies the engine with the
                model.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.handlers import ModelFreezer
            >>> from gravitorch.testing import create_dummy_engine
            >>> engine = create_dummy_engine()
            >>> engine.model
            DummyClassificationModel(
              (linear): Linear(in_features=4, out_features=3, bias=True)
              (criterion): CrossEntropyLoss()
            )
            >>> handler = ModelFreezer()
            >>> handler.freeze(engine)
            >>> for name, param in engine.model.named_parameters():
            ...     print(name, param.requires_grad)
            ...
            linear.weight False
            linear.bias False
        """
        if self._module_name:
            logger.info(f"Freeze submodule {self._module_name}")
        else:
            logger.info("Freeze model")
        freeze_module(engine.model.get_submodule(self._module_name))
