r"""This module implements an evaluation loop using automatic mixed precision
(AMP)."""

__all__ = ["AMPEvaluationLoop"]

import logging
from typing import Any, Union

import torch
from torch.cuda.amp import autocast
from torch.nn import Module

from gravitorch.engines.base import BaseEngine
from gravitorch.engines.events import EngineEvents
from gravitorch.loops.evaluation.conditions import BaseEvalCondition
from gravitorch.loops.evaluation.vanilla import VanillaEvaluationLoop
from gravitorch.loops.observers.base import BaseLoopObserver
from gravitorch.utils.device_placement import BaseDevicePlacement
from gravitorch.utils.profilers.base import BaseProfiler

logger = logging.getLogger(__name__)


class AMPEvaluationLoop(VanillaEvaluationLoop):
    r"""Implements a training loop to train a model on a dataset by using
    training loop using automatic mixed precision (AMP).

    Args:
        grad_enabled (bool, optional): Specifies if the gradient is
            computed or not in the evaluation loop. By default, the
            gradient is not computed to reduce the memory footprint.
            Default: ``False``
        amp_enabled (bool, optional): If ``True``, automatic mixed
            precision (AMP) is enabled, otherwise it is disabled.
            Default: ``True``
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
        grad_enabled: bool = False,
        amp_enabled: bool = True,
        batch_device_placement: Union[BaseDevicePlacement, dict, None] = None,
        condition: Union[BaseEvalCondition, dict, None] = None,
        observer: Union[BaseLoopObserver, dict, None] = None,
        profiler: Union[BaseProfiler, dict, None] = None,
    ):
        super().__init__(
            tag=tag,
            grad_enabled=grad_enabled,
            batch_device_placement=batch_device_placement,
            condition=condition,
            observer=observer,
            profiler=profiler,
        )
        self._amp_enabled = bool(amp_enabled)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  tag={self._tag},\n"
            f"  batch_device_placement={self._batch_device_placement},\n"
            f"  grad_enabled={self._grad_enabled},\n"
            f"  amp_enabled={self._amp_enabled},\n"
            f"  condition={self._condition},\n"
            f"  observer={self._observer},\n"
            f"  profiler={self._profiler},\n"
            ")"
        )

    def _eval_one_batch(self, engine: BaseEngine, model: Module, batch: Any) -> dict:
        engine.fire_event(EngineEvents.EVAL_ITERATION_STARTED)
        with torch.set_grad_enabled(self._grad_enabled):
            with autocast(enabled=self._amp_enabled):
                output = model(self._batch_device_placement.send(batch))
        engine.fire_event(EngineEvents.EVAL_ITERATION_COMPLETED)
        return output
