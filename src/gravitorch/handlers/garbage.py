from __future__ import annotations

__all__ = ["EpochGarbageCollector"]

import gc
import logging
from typing import TYPE_CHECKING

from gravitorch.engines.events import EngineEvents, EpochPeriodicCondition
from gravitorch.handlers.base import BaseHandler
from gravitorch.handlers.utils import add_unique_event_handler
from gravitorch.utils.events import GConditionalEventHandler

if TYPE_CHECKING:
    from gravitorch.engines import BaseEngine

logger = logging.getLogger(__name__)


class EpochGarbageCollector(BaseHandler):
    r"""Implements a handler to perform garbage collection.

    Args:
    ----
        event (str, optional): Specifies the epoch-based event when
            the garbage collection should be performed.
            Default: ``'epoch_completed'``
        freq (int, optional): Specifies the epoch frequency used to
            perform garbage collection. Default: ``1``

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.handlers import EpochGarbageCollector
        >>> from gravitorch.testing import create_dummy_engine
        >>> engine = create_dummy_engine()
        >>> handler = EpochGarbageCollector()
        >>> handler
        EpochGarbageCollector(freq=1, event=epoch_completed)
        >>> handler.attach(engine)
        >>> engine.trigger_event("epoch_completed")
    """

    def __init__(self, event: str = EngineEvents.EPOCH_COMPLETED, freq: int = 1) -> None:
        self._event = str(event)
        if freq < 1:
            raise ValueError(f"freq has to be greater than 0 (received: {freq:,})")
        self._freq = int(freq)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(freq={self._freq}, event={self._event})"

    def attach(self, engine: BaseEngine) -> None:
        add_unique_event_handler(
            engine=engine,
            event=self._event,
            event_handler=GConditionalEventHandler(
                self.collect,
                condition=EpochPeriodicCondition(engine=engine, freq=self._freq),
                handler_kwargs={"engine": engine},
            ),
        )

    def collect(self, engine: BaseEngine) -> None:
        r"""Performs garbage collection.

        Args:
        ----
            engine (``BaseEngine``): Specifies the engine.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.handlers import EpochGarbageCollector
            >>> from gravitorch.testing import create_dummy_engine
            >>> engine = create_dummy_engine()
            >>> handler = EpochGarbageCollector()
            >>> handler.collect(engine)
        """
        gc.collect()
