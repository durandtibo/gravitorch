r"""This module defines the base class for the evaluation loops."""

__all__ = ["BaseBasicEvaluationLoop"]

import logging
import sys
from abc import abstractmethod
from collections.abc import Iterable
from typing import Any, Union

from torch.nn import Module

from gravitorch import constants as ct
from gravitorch.distributed import comm as dist
from gravitorch.engines.base import BaseEngine
from gravitorch.engines.events import EngineEvents
from gravitorch.loops.evaluation.base import BaseEvaluationLoop
from gravitorch.loops.evaluation.conditions import (
    BaseEvalCondition,
    EveryEpochEvalCondition,
)
from gravitorch.loops.observers import BaseLoopObserver, setup_loop_observer
from gravitorch.utils.history import MinScalarHistory
from gravitorch.utils.metric_tracker import ScalarMetricTracker
from gravitorch.utils.profilers import BaseProfiler, setup_profiler
from gravitorch.utils.seed import manual_seed
from gravitorch.utils.timing import BatchLoadingTimer

logger = logging.getLogger(__name__)


class BaseBasicEvaluationLoop(BaseEvaluationLoop):
    r"""Implements a simple evaluation loop to evaluate a model on a given
    dataset.

    Args:
        tag (str, optional): Specifies the tag which is used to log
            metrics. Default: ``"eval"``
        condition (``BaseEvalCondition`` or dict or None): Specifies
            the condition to evaluate the loop or its configuration.
            If ``None``, the ``EveryEpochEvalCondition(every=1)`` is
            used.  Default ``None``
        observer (``BaseLoopObserver`` or dict or None, optional):
            Specifies the loop observer or its configuration.
            If ``None``, the ``NoOpLoopObserver`` is instantiated.
            Default: ``None``
        profiler (``BaseProfiler`` or dict or None, optional):
            Specifies the profiler or its configuration. If ``None``,
            the ``NoOpProfiler`` is instantiated. Default: ``None``
    """

    def __init__(
        self,
        tag: str = "eval",
        condition: Union[BaseEvalCondition, dict, None] = None,
        observer: Union[BaseLoopObserver, dict, None] = None,
        profiler: Union[BaseProfiler, dict, None] = None,
    ):
        self._tag = str(tag)
        self._condition = self._setup_condition(condition)
        logger.info(f"condition:\n{self._condition}")
        self._observer = setup_loop_observer(observer)
        logger.info(f"observer:\n{self._observer}")
        self._profiler = setup_profiler(profiler)
        logger.info(f"profiler:\n{self._profiler}")

    def eval(self, engine: BaseEngine) -> None:
        r"""Evaluates the model on the evaluation dataset.

        Args:
            engine (``BaseEngine``): Specifies the engine.
        """
        dist.barrier()
        if not engine.data_source.has_data_loader(self._tag) or not self._condition(engine):
            return
        logger.info(f"Evaluating model for epoch {engine.epoch}")

        self._prepare_evaluation(engine)
        engine.fire_event(EngineEvents.EVAL_EPOCH_STARTED)

        model, data_loader = self._prepare_model_data_loader(engine)

        # Evaluate the model on each mini-match in the dataset.
        metrics = ScalarMetricTracker()
        data_loader = BatchLoadingTimer(data_loader, epoch=engine.epoch, prefix=f"{self._tag}/")
        self._observer.start(engine)
        dist.barrier()

        with self._profiler as profiler:
            for batch in data_loader:
                # Run forward on the given batch.
                output = self._eval_one_batch(engine, model, batch)
                metrics.update(output)
                self._observer.update(engine=engine, model_input=batch, model_output=output)
                profiler.step()

        # To be sure the progress bar is displayed before the following lines
        sys.stdout.flush()
        dist.barrier()
        self._observer.end(engine)

        # Log some evaluation metrics to the engine.
        data_loader.log_stats(engine=engine)
        metrics.log_average_value(engine=engine, prefix=f"{self._tag}/")
        dist.barrier()

        engine.fire_event(EngineEvents.EVAL_EPOCH_COMPLETED)

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        pass

    def state_dict(self) -> dict[str, Any]:
        return {}

    def _prepare_evaluation(self, engine: BaseEngine) -> None:
        r"""Prepares the evaluation.

        Args:
            engine (``BaseEngine``): Specifies the engine.
        """
        # Fix the random seed for reproducibility purpose.
        manual_seed(engine.random_seed + engine.epoch + engine.max_epochs * dist.get_rank())
        engine.model.eval()

        if not engine.has_history(f"{self._tag}/{ct.LOSS}"):
            engine.add_history(MinScalarHistory(f"{self._tag}/{ct.LOSS}"))

    def _setup_condition(
        self, condition: Union[BaseEvalCondition, dict, None]
    ) -> BaseEvalCondition:
        r"""Sets up the condition.

        The condition is instantiated from its configuration by using
        the ``BaseEvalCondition`` factory function.

        Args:
            condition (``BaseEvalCondition`` or dict or None):
                Specifies the condition or its configuration.
                If ``None``, the ``EveryEpochEvalCondition(every=1)``
                is instantiated.

        Returns:
            ``BaseEvalCondition``: The state.
        """
        condition = condition or EveryEpochEvalCondition(every=1)
        if isinstance(condition, dict):
            logger.info("Initializing the condition from its configuration...")
            condition = BaseEvalCondition.factory(**condition)

        logger.info(f"condition:\n{condition}")
        return condition

    @abstractmethod
    def _eval_one_batch(self, engine: BaseEngine, model: Module, batch: Any) -> dict:
        r"""Evaluates the model on the given batch.

        Args:
            engine (``BaseEngine``): Specifies the engine.
            model (``torch.nn.Module``): Specifies the model to evaluate.
            batch: Specifies the batch of data.

        Returns:
            dict: Some results about the batch.
        """

    @abstractmethod
    def _prepare_model_data_loader(self, engine: BaseEngine) -> tuple[Module, Iterable]:
        r"""Prepares the model, optimizer and data loader.

        Args:
            engine (``BaseEngine``): Specifies the engine.

        Returns:
            ``torch.nn.Module``, ``Iterable``: A tuple with the model
                and the data loader.
        """
