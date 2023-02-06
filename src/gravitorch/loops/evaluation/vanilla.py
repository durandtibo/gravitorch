r"""This module implements a simple evaluation loop."""

__all__ = ["VanillaEvaluationLoop"]

import logging
import sys
from collections.abc import Iterable
from typing import Any, Union

import torch
from torch.nn import Module
from tqdm import tqdm

from gravitorch.distributed import comm as dist
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


class VanillaEvaluationLoop(BaseBasicEvaluationLoop):
    r"""Implements a simple evaluation loop to evaluate a model on a given
    dataset.

    Args:
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
    """

    def __init__(
        self,
        grad_enabled: bool = False,
        batch_device_placement: Union[BaseDevicePlacement, dict, None] = None,
        tag: str = "eval",
        condition: Union[BaseEvalCondition, dict, None] = None,
        observer: Union[BaseLoopObserver, dict, None] = None,
        profiler: Union[BaseProfiler, dict, None] = None,
    ):
        super().__init__(tag=tag, condition=condition, observer=observer, profiler=profiler)
        self._grad_enabled = bool(grad_enabled)
        self._batch_device_placement = setup_device_placement(
            batch_device_placement or AutoDevicePlacement()
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  tag={self._tag},\n"
            f"  batch_device_placement={self._batch_device_placement},\n"
            f"  grad_enabled={self._grad_enabled},\n"
            f"  condition={self._condition},\n"
            f"  observer={self._observer},\n"
            f"  profiler={self._profiler},\n"
            ")"
        )

    def _eval_one_batch(self, engine: BaseEngine, model: Module, batch: Any) -> dict:
        engine.fire_event(EngineEvents.EVAL_ITERATION_STARTED)
        with torch.set_grad_enabled(self._grad_enabled):
            output = model(self._batch_device_placement.send(batch))
        engine.fire_event(EngineEvents.EVAL_ITERATION_COMPLETED)
        return output

    def _prepare_model_data_loader(self, engine: BaseEngine) -> tuple[Module, Iterable]:
        logger.info("Preparing the model and data loader...")
        data_loader = engine.data_source.get_data_loader(loader_id=self._tag, engine=engine)
        prefix = f"({dist.get_rank()}/{dist.get_world_size()}) " if dist.is_distributed() else ""
        data_loader = tqdm(
            data_loader,
            desc=f"{prefix}Evaluation [{engine.epoch}]",
            position=dist.get_rank(),
            file=sys.stdout,
        )
        logger.info("Evaluation data loader has been created")
        return engine.model, data_loader
