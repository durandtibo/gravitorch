__all__ = ["EpochCudaMemoryMonitor", "IterationCudaMemoryMonitor"]

import torch

from gravitorch.engines.base import BaseEngine
from gravitorch.engines.events import EngineEvents
from gravitorch.handlers.base import BaseHandler
from gravitorch.handlers.utils import add_unique_event_handler
from gravitorch.utils.cudamem import log_max_cuda_memory_allocated
from gravitorch.utils.events import (
    ConditionalEventHandler,
    EpochPeriodicCondition,
    IterationPeriodicCondition,
)
from gravitorch.utils.exp_trackers import EpochStep, IterationStep


class EpochCudaMemoryMonitor(BaseHandler):
    r"""Implements a handler to monitor the CUDA memory usage every ``freq``
    epochs.

    Args:
        event (str, optional): Specifies the epoch-based event when
            the CUDA memory usage should be capture.
            Default: ``'epoch_completed'``
        freq (int, optional): Specifies the epoch frequency used to
            monitor the CUDA memory usage. Default: ``1``
    """

    def __init__(self, event: str = EngineEvents.EPOCH_COMPLETED, freq: int = 1):
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
            event_handler=ConditionalEventHandler(
                self.monitor,
                condition=EpochPeriodicCondition(engine=engine, freq=self._freq),
                handler_kwargs={"engine": engine},
            ),
        )

    def monitor(self, engine: BaseEngine) -> None:
        r"""Monitors the CUDA memory usage.

        Args:
            engine (``BaseEngine``): Specifies the engine.
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
    r"""Implements a handler to monitor the CUDA memory usage every ``freq``
    iterations.

    Args:
        event (str, optional): Specifies the iteration-based event
            when the CUDA memory usage should be capture.
            Default: ``'epoch_completed'``
        freq (int, optional): Specifies the iteration frequency used
            to monitor the CUDA memory usage. Default: ``1``
    """

    def __init__(self, event: str = EngineEvents.TRAIN_ITERATION_COMPLETED, freq: int = 1):
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
            event_handler=ConditionalEventHandler(
                self.monitor,
                condition=IterationPeriodicCondition(engine=engine, freq=self._freq),
                handler_kwargs={"engine": engine},
            ),
        )

    def monitor(self, engine: BaseEngine) -> None:
        r"""Monitors the CUDA memory usage.

        Args:
            engine (``BaseEngine``): Specifies the engine.
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
