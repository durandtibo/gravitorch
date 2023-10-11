from __future__ import annotations

__all__ = ["ConsolidateOptimizerState"]

import logging
from typing import TYPE_CHECKING

from gravitorch import distributed as dist
from gravitorch.engines.events import EngineEvents
from gravitorch.handlers.base import BaseHandler
from gravitorch.handlers.utils import add_unique_event_handler
from gravitorch.utils.events import GEventHandler

if TYPE_CHECKING:
    from gravitorch.engines import BaseEngine

logger = logging.getLogger(__name__)


class ConsolidateOptimizerState(BaseHandler):
    r"""Implements a handler to consolidate the state dict of an
    optimizer.

    Args:
    ----
        event (str, optional): Specifies the event when the optimizer
            state dict is consolidated.
            Default: ``'train_epoch_completed'``
        recipient_rank (int, optional): Specifies on which rank to
            materialize the full state dict. ``-1`` is a special
            value, which means that all ranks should have the state.
            Default: ``0``

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.handlers import ConsolidateOptimizerState
        >>> from gravitorch.testing import create_dummy_engine
        >>> engine = create_dummy_engine()
        >>> handler = ConsolidateOptimizerState()
        >>> handler
        ConsolidateOptimizerState(event=train_epoch_completed, recipient_rank=0)
        >>> handler.attach(engine)
        >>> engine.trigger_event("train_epoch_completed")
    """

    def __init__(
        self, event: str = EngineEvents.TRAIN_EPOCH_COMPLETED, recipient_rank: int = 0
    ) -> None:
        self._event = str(event)
        self._recipient_rank = int(recipient_rank)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(event={self._event}, "
            f"recipient_rank={self._recipient_rank})"
        )

    def attach(self, engine: BaseEngine) -> None:
        r"""Attaches the event handler to the engine.

        Args:
        ----
            engine (``BaseEngine``): Specifies the engine.
        """
        add_unique_event_handler(
            engine=engine,
            event=self._event,
            event_handler=GEventHandler(
                self.consolidate,
                handler_kwargs={"engine": engine},
            ),
        )

    def consolidate(self, engine: BaseEngine) -> None:
        r"""Consolidate the optimizer state dict.

        Args:
        ----
            engine (``BaseEngine``): Specifies the engine.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.handlers import ConsolidateOptimizerState
            >>> from gravitorch.testing import create_dummy_engine
            >>> engine = create_dummy_engine()
            >>> handler = ConsolidateOptimizerState()
            >>> handler.consolidate(engine)
        """
        if not engine.optimizer:
            logger.info(
                "It is not possible to consolidate the optimizer state dict "
                "because there is no optimizer"
            )
        elif hasattr(engine.optimizer, "consolidate_state_dict"):
            logger.info(f"Consolidating the optimizer state dict on rank {self._recipient_rank}...")
            engine.optimizer.consolidate_state_dict(self._recipient_rank)
            dist.barrier()
        else:
            logger.info("The optimizer does not have a 'consolidate_state_dict' method")
