r"""This module implements a simple evaluation loop."""

from __future__ import annotations

__all__ = ["EvaluationLoop"]

import logging
from collections.abc import Iterable
from typing import Any

import torch
from coola.utils import str_indent, str_mapping
from torch.nn import Module

from gravitorch.engines.base import BaseEngine
from gravitorch.engines.events import EngineEvents
from gravitorch.loops.evaluation.basic import BaseBasicEvaluationLoop
from gravitorch.loops.evaluation.conditions import BaseEvalCondition
from gravitorch.loops.observers import BaseLoopObserver
from gravitorch.utils.device_placement import (
    AutoDevicePlacement,
    BaseDevicePlacement,
    setup_device_placement,
)
from gravitorch.utils.profilers import BaseProfiler

logger = logging.getLogger(__name__)


class EvaluationLoop(BaseBasicEvaluationLoop):
    r"""Implements a simple evaluation loop to evaluate a model on a
    given dataset.

    Args:
    ----
        grad_enabled (bool, optional): Specifies if the gradient is
            computed or not in the evaluation loop. By default, the
            gradient is not computed to reduce the memory footprint.
            Default: ``False``
        batch_device_placement (``BaseDevicePlacement`` or dict or
            ``None``, optional): Specifies the batch device placement
            module. This module moves the batch on a target device.
            The target device should be compatible with the model.
            If ``None``, an ``AutoDevicePlacement`` object is
            instantiated. Default: ``None``
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

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.loops.evaluation import EvaluationLoop
        >>> from gravitorch.testing import create_dummy_engine
        >>> engine = create_dummy_engine()
        >>> loop = EvaluationLoop()
        >>> loop
        EvaluationLoop(
          (batch_device_placement): AutoDevicePlacement(device=cpu)
          (grad_enabled): False
          (tag): eval
          (condition): EveryEpochEvalCondition(every=1)
          (observer): NoOpLoopObserver()
          (profiler): NoOpProfiler()
        )
        >>> loop.eval(engine)
    """

    def __init__(
        self,
        grad_enabled: bool = False,
        batch_device_placement: BaseDevicePlacement | dict | None = None,
        tag: str = "eval",
        condition: BaseEvalCondition | dict | None = None,
        observer: BaseLoopObserver | dict | None = None,
        profiler: BaseProfiler | dict | None = None,
    ) -> None:
        super().__init__(tag=tag, condition=condition, observer=observer, profiler=profiler)
        self._grad_enabled = bool(grad_enabled)
        self._batch_device_placement = setup_device_placement(
            batch_device_placement or AutoDevicePlacement()
        )

    def __repr__(self) -> str:
        args = str_indent(
            str_mapping(
                {
                    "batch_device_placement": self._batch_device_placement,
                    "grad_enabled": self._grad_enabled,
                    "tag": self._tag,
                    "condition": self._condition,
                    "observer": self._observer,
                    "profiler": self._profiler,
                }
            )
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def _eval_one_batch(self, engine: BaseEngine, model: Module, batch: Any) -> dict:
        engine.trigger_event(EngineEvents.EVAL_ITERATION_STARTED)
        with torch.set_grad_enabled(self._grad_enabled):
            output = model(self._batch_device_placement.send(batch))
        engine.trigger_event(EngineEvents.EVAL_ITERATION_COMPLETED)
        return output

    def _prepare_model_dataiter(self, engine: BaseEngine) -> tuple[Module, Iterable]:
        logger.info("Preparing the model and data iterable...")
        dataiter = engine.datasource.get_iterable(iter_id=self._tag, engine=engine)
        logger.info("Evaluation model and data iterable have been prepared")
        return engine.model, dataiter
