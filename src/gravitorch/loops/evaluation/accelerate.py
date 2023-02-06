r"""This module implements an evaluation loop using the Accelerate library
(https://huggingface.co/docs/accelerate)."""

__all__ = ["AccelerateEvaluationLoop"]

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
from gravitorch.utils.integrations import check_accelerate, is_accelerate_available
from gravitorch.utils.profilers import BaseProfiler

if is_accelerate_available():
    from accelerate import Accelerator
else:
    Accelerator = None  # pragma: no cover

logger = logging.getLogger(__name__)


class AccelerateEvaluationLoop(BaseBasicEvaluationLoop):
    r"""Implements an evaluation loop that uses ``accelerate.Accelerator`` to
    evaluate a model.

    Args:
        accelerator (``accelerate.Accelerate`` or dict or None,
            optional): Specifies the ``accelerate.Accelerate`` object
            or the parameters to instantiate it. Please read the
            ``accelerate.Accelerator`` documentation to know the
            parameters https://huggingface.co/docs/accelerate/accelerator.html.
            If ``None``, it will use the default parameters.
            Default: ``None``
        tag (str, optional): Specifies the tag which is used to log
            metrics. Default: ``"eval"``
        grad_enabled (bool, optional): Specifies if the gradient is
            computed or not in the evaluation loop. By default, the
            gradient is not computed to reduce the memory footprint.
            Default: ``False``
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
        accelerator: Union[Accelerator, dict, None] = None,
        tag: str = "eval",
        grad_enabled: bool = False,
        condition: Union[BaseEvalCondition, dict, None] = None,
        observer: Union[BaseLoopObserver, dict, None] = None,
        profiler: Union[BaseProfiler, dict, None] = None,
    ):
        check_accelerate()
        self._accelerator = self._setup_accelerator(accelerator or {})
        logger.info(f"accelerator state:\n{self._accelerator.state}")
        super().__init__(tag=tag, condition=condition, observer=observer, profiler=profiler)
        self._grad_enabled = grad_enabled

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  tag={self._tag},\n"
            f"  grad_enabled={self._grad_enabled},\n"
            f"  condition={self._condition},\n"
            f"  observer={self._observer},\n"
            f"  profiler={self._profiler},\n"
            ")"
        )

    def _eval_one_batch(self, engine: BaseEngine, model: Module, batch: Any) -> dict:
        engine.fire_event(EngineEvents.EVAL_ITERATION_STARTED)
        with torch.set_grad_enabled(self._grad_enabled):
            output = model(batch)
        engine.fire_event(EngineEvents.EVAL_ITERATION_COMPLETED)
        return output

    def _prepare_model_data_loader(self, engine: BaseEngine) -> tuple[Module, Iterable]:
        logger.info("Preparing the model and data loader...")
        model, data_loader = self._accelerator.prepare(
            [
                engine.model,
                engine.data_source.get_data_loader(loader_id=self._tag, engine=engine),
            ]
        )

        prefix = f"({dist.get_rank()}/{dist.get_world_size()}) " if dist.is_distributed() else ""
        data_loader = tqdm(
            data_loader,
            desc=f"{prefix}Evaluation [{engine.epoch}]",
            position=dist.get_rank(),
            file=sys.stdout,
        )
        logger.info("Evaluation data loader has been created")
        return model, data_loader

    def _setup_accelerator(self, accelerator: Union[Accelerator, dict]) -> Accelerator:
        r"""Sets up the accelerator.

        Args:
            accelerator (``accelerate.Accelerator`` or dict, optional):
                Specifies the ``accelerate.Accelerator`` object or the
                parameters to instantiate it. Please read the
                ``accelerate.Accelerator`` documentation to know the
                parameters https://huggingface.co/docs/accelerate/accelerator.html.

        Returns:
            ``accelerate.Accelerator``: The accelerator object.

        Raises:
            RuntimeError: if the accelerate package is not installed.
        """
        if isinstance(accelerator, Accelerator):
            return accelerator
        logger.info(f"accelerator options: {accelerator}")
        return Accelerator(**accelerator)
