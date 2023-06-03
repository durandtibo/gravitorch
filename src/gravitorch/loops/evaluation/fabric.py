r"""This module implements an evaluation loop using the Fabric library
(https://huggingface.co/docs/accelerate)."""

__all__ = ["FabricEvaluationLoop"]

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
from gravitorch.utils.imports import check_lightning, is_lightning_available
from gravitorch.utils.profilers import BaseProfiler

if is_lightning_available():
    from lightning import Fabric
    from lightning.fabric.accelerators import ACCELERATOR_REGISTRY
else:
    Fabric = None  # pragma: no cover

logger = logging.getLogger(__name__)


class FabricEvaluationLoop(BaseBasicEvaluationLoop):
    r"""Implements an evaluation loop that uses ``accelerate.Fabric`` to
    evaluate a model.

    Args:
    ----
        fabric (``lightning.Fabric`` or dict or None, optional):
            Specifies the ``lightning.Fabric`` object or the
            parameters to instantiate it. Please read the
            ``lightning.Fabric`` documentation to know the
            parameters
            https://lightning.ai/docs/fabric/stable/.
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
        fabric: Union[Fabric, dict, None] = None,
        tag: str = "eval",
        grad_enabled: bool = False,
        condition: Union[BaseEvalCondition, dict, None] = None,
        observer: Union[BaseLoopObserver, dict, None] = None,
        profiler: Union[BaseProfiler, dict, None] = None,
    ) -> None:
        check_lightning()
        self._fabric = self._setup_fabric(fabric or {})
        super().__init__(tag=tag, condition=condition, observer=observer, profiler=profiler)
        self._grad_enabled = grad_enabled

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  fabric={self._fabric},\n"
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
        logger.info("Preparing the model...")
        model = self._fabric.setup(engine.model)
        logger.info("Model is ready for evaluation")

        logger.info("Preparing the data loader...")
        data_loader = engine.data_source.get_data_loader(loader_id=self._tag, engine=engine)
        data_loader = self._fabric.setup_dataloaders(data_loader)
        prefix = f"({dist.get_rank()}/{dist.get_world_size()}) " if dist.is_distributed() else ""
        data_loader = tqdm(
            data_loader,
            desc=f"{prefix}Evaluation [{engine.epoch}]",
            position=dist.get_rank(),
            file=sys.stdout,
        )
        logger.info("Evaluation data loader is ready")
        return model, data_loader

    def _setup_fabric(self, fabric: Union[Fabric, dict]) -> Fabric:
        r"""Sets up the ``lightning.Fabric`` object.

        Args:
        ----
            fabric (``lightning.Fabric`` or dict or None, optional):
                Specifies the ``lightning.Fabric`` object or the
                parameters to instantiate it. Please read the
                ``lightning.Fabric`` documentation to know the
                parameters
                https://lightning.ai/docs/fabric/stable/.
                If ``None``, it will use the default parameters.

        Returns:
        -------
            ``lightning.Fabric``: The instantiated
                ``lightning.Fabric`` object.

        Raises:
        ------
            RuntimeError: if the ``lightning`` package is not
                installed.
        """
        if isinstance(fabric, Fabric):
            return fabric
        logger.info(f"Available accelerators: {ACCELERATOR_REGISTRY.available_accelerators()}")
        logger.info(f"Fabric configuration: {fabric}")
        return Fabric(**fabric)
