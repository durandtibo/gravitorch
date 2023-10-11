from __future__ import annotations

__all__ = [
    "EpochCudaEmptyCache",
    "EpochCudaMemoryMonitor",
    "IterationCudaEmptyCache",
    "IterationCudaMemoryMonitor",
]

import logging
from typing import TYPE_CHECKING

import torch

from gravitorch.engines.events import (
    EngineEvents,
    EpochPeriodicCondition,
    IterationPeriodicCondition,
)
from gravitorch.handlers.base import BaseHandler
from gravitorch.handlers.utils import add_unique_event_handler
from gravitorch.utils.cudamem import log_max_cuda_memory_allocated
from gravitorch.utils.events import GConditionalEventHandler
from gravitorch.utils.exp_trackers import EpochStep, IterationStep

if TYPE_CHECKING:
    from gravitorch.engines import BaseEngine

logger = logging.getLogger(__name__)


class EpochCudaMemoryMonitor(BaseHandler):
    r"""Implements a handler to monitor the CUDA memory usage every
    ``freq`` epochs.

    Args:
    ----
        event (str, optional): Specifies the epoch-based event when
            the CUDA memory usage should be captured.
            Default: ``'epoch_completed'``
        freq (int, optional): Specifies the epoch frequency used to
            monitor the CUDA memory usage. Default: ``1``

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.handlers import EpochCudaMemoryMonitor
        >>> from gravitorch.testing import create_dummy_engine
        >>> engine = create_dummy_engine()
        >>> handler = EpochCudaMemoryMonitor()
        >>> handler
        EpochCudaMemoryMonitor(freq=1, event=epoch_completed)
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
                self.monitor,
                condition=EpochPeriodicCondition(engine=engine, freq=self._freq),
                handler_kwargs={"engine": engine},
            ),
        )

    def monitor(self, engine: BaseEngine) -> None:
        r"""Monitors the CUDA memory usage.

        Args:
        ----
            engine (``BaseEngine``): Specifies the engine.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.handlers import EpochCudaMemoryMonitor
            >>> from gravitorch.testing import create_dummy_engine
            >>> engine = create_dummy_engine()
            >>> handler = EpochCudaMemoryMonitor()
            >>> handler.monitor(engine)
        """
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            log_max_cuda_memory_allocated()
            allocated_memory = torch.cuda.max_memory_allocated()
            total_memory = torch.cuda.mem_get_info()[1]
            engine.log_metrics(
                {
                    "epoch/max_cuda_memory_allocated": allocated_memory,
                    "epoch/max_cuda_memory_allocated_pct": float(allocated_memory / total_memory),
                },
                step=EpochStep(engine.epoch),
            )


class IterationCudaMemoryMonitor(BaseHandler):
    r"""Implements a handler to monitor the CUDA memory usage every
    ``freq`` iterations.

    Args:
    ----
        event (str, optional): Specifies the iteration-based event
            when the CUDA memory usage should be capture.
            Default: ``'epoch_completed'``
        freq (int, optional): Specifies the iteration frequency used
            to monitor the CUDA memory usage. Default: ``1``

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.handlers import IterationCudaMemoryMonitor
        >>> from gravitorch.testing import create_dummy_engine
        >>> engine = create_dummy_engine()
        >>> handler = IterationCudaMemoryMonitor()
        >>> handler
        IterationCudaMemoryMonitor(freq=1, event=train_iteration_completed)
        >>> handler.attach(engine)
        >>> engine.trigger_event("train_iteration_completed")
    """

    def __init__(self, event: str = EngineEvents.TRAIN_ITERATION_COMPLETED, freq: int = 1) -> None:
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
                self.monitor,
                condition=IterationPeriodicCondition(engine=engine, freq=self._freq),
                handler_kwargs={"engine": engine},
            ),
        )

    def monitor(self, engine: BaseEngine) -> None:
        r"""Monitors the CUDA memory usage.

        Args:
        ----
            engine (``BaseEngine``): Specifies the engine.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.handlers import IterationCudaMemoryMonitor
            >>> from gravitorch.testing import create_dummy_engine
            >>> engine = create_dummy_engine()
            >>> handler = IterationCudaMemoryMonitor()
            >>> handler.monitor(engine)
        """
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            log_max_cuda_memory_allocated()
            allocated_memory = torch.cuda.max_memory_allocated()
            total_memory = torch.cuda.mem_get_info()[1]
            engine.log_metrics(
                {
                    "iteration/max_cuda_memory_allocated": allocated_memory,
                    "iteration/max_cuda_memory_allocated_pct": float(
                        allocated_memory / total_memory
                    ),
                },
                step=IterationStep(engine.iteration),
            )


class EpochCudaEmptyCache(BaseHandler):
    r"""Implements a handler to empty the CUDA cache every ``freq``
    epochs.

    Args:
    ----
        event (str, optional): Specifies the epoch-based event when
            the learning rate should be capture.
            Default: ``'epoch_completed'``
        freq (int, optional): Specifies the epoch frequency used to
            monitor the learning rate. Default: ``1``

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.handlers import EpochCudaEmptyCache
        >>> from gravitorch.testing import create_dummy_engine
        >>> engine = create_dummy_engine()
        >>> handler = EpochCudaEmptyCache()
        >>> handler
        EpochCudaEmptyCache(freq=1, event=epoch_completed)
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
                self.empty_cache,
                condition=EpochPeriodicCondition(engine=engine, freq=self._freq),
                handler_kwargs={"engine": engine},
            ),
        )

    def empty_cache(self, engine: BaseEngine) -> None:
        r"""Empty the CUDA cache.

        Args:
        ----
            engine (``BaseEngine``): Specifies the engine.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.handlers import EpochCudaEmptyCache
            >>> from gravitorch.testing import create_dummy_engine
            >>> engine = create_dummy_engine()
            >>> handler = EpochCudaEmptyCache()
            >>> handler.empty_cache(engine)
        """
        if torch.cuda.is_available():
            logger.info("Emptying CUDA cache...")
            torch.cuda.empty_cache()


class IterationCudaEmptyCache(BaseHandler):
    r"""Implements a handler to empty the CUDA cache every ``freq``
    iterations.

    Args:
    ----
        event (str, optional): Specifies the iteration-based event when
            the learning rate should be capture.
            Default: ``'train_iteration_completed'``
        freq (int, optional): Specifies the iteration frequency used to
            monitor the learning rate. Default: ``1``

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.handlers import IterationCudaEmptyCache
        >>> from gravitorch.testing import create_dummy_engine
        >>> engine = create_dummy_engine()
        >>> handler = IterationCudaEmptyCache()
        >>> handler
        IterationCudaEmptyCache(freq=1, event=train_iteration_completed)
        >>> handler.attach(engine)
        >>> engine.trigger_event("train_iteration_completed")
    """

    def __init__(self, event: str = EngineEvents.TRAIN_ITERATION_COMPLETED, freq: int = 1) -> None:
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
                self.empty_cache,
                condition=IterationPeriodicCondition(engine=engine, freq=self._freq),
                handler_kwargs={"engine": engine},
            ),
        )

    def empty_cache(self, engine: BaseEngine) -> None:
        r"""Empty the CUDA cache.

        Args:
        ----
            engine (``BaseEngine``): Specifies the engine.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.handlers import IterationCudaEmptyCache
            >>> from gravitorch.testing import create_dummy_engine
            >>> engine = create_dummy_engine()
            >>> handler = IterationCudaEmptyCache()
            >>> handler.empty_cache(engine)
        """
        if torch.cuda.is_available():
            logger.info("Emptying CUDA cache...")
            torch.cuda.empty_cache()
