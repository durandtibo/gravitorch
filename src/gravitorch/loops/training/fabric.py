__all__ = ["FabricTrainingLoop"]

import logging
import sys
from collections.abc import Callable, Iterable
from typing import Any, Optional, Union

import torch
from torch.nn import Module
from torch.optim import Optimizer
from tqdm import tqdm

from gravitorch import constants as ct
from gravitorch.distributed import comm as dist
from gravitorch.engines.base import BaseEngine
from gravitorch.engines.events import EngineEvents
from gravitorch.loops.observers import BaseLoopObserver
from gravitorch.loops.training.basic import BaseBasicTrainingLoop
from gravitorch.loops.training.utils import setup_clip_grad
from gravitorch.utils.imports import check_lightning, is_lightning_available
from gravitorch.utils.profilers import BaseProfiler

if is_lightning_available():
    from lightning import Fabric
    from lightning.fabric.accelerators import ACCELERATOR_REGISTRY
else:
    Fabric = None  # pragma: no cover

logger = logging.getLogger(__name__)


class FabricTrainingLoop(BaseBasicTrainingLoop):
    r"""Implements a training loop that uses ``lightning.Fabric`` to train
    a model.

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
        set_grad_to_none (bool, optional): If ``True``, set the
            gradients to ``None``, otherwise set the gradients to
            zero. Setting the gradients to ``None`` will in general
            have lower memory footprint, and can modestly improve
            performance. Default: ``True``
        tag (str, optional): Specifies the tag which is used to log
            metrics. Default: ``"train"``
        clip_grad (dict or None, optional): Specifies the
            configuration to clip the gradient. If ``None``, no
            gradient clipping is used during the training.
            Default: ``None``
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
        set_grad_to_none: bool = True,
        tag: str = "train",
        clip_grad: Optional[dict] = None,
        observer: Union[BaseLoopObserver, dict, None] = None,
        profiler: Union[BaseProfiler, dict, None] = None,
    ) -> None:
        check_lightning()
        self._fabric = self._setup_fabric(fabric or {})
        super().__init__(tag=tag, clip_grad=clip_grad, observer=observer, profiler=profiler)
        self._set_grad_to_none = bool(set_grad_to_none)
        self._fabric.launch()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  fabric={self._fabric},\n"
            f"  set_grad_to_none={self._set_grad_to_none},\n"
            f"  tag={self._tag},\n"
            f"  clip_grad_fn={self._clip_grad_fn},\n"
            f"  clip_grad_args={self._clip_grad_args},\n"
            f"  observer={self._observer},\n"
            f"  profiler={self._profiler},\n"
            ")"
        )

    def _prepare_model_optimizer_data_loader(
        self, engine: BaseEngine
    ) -> tuple[Module, Optimizer, Iterable]:
        r"""Prepares the model, optimizer and data loader.

        Args:
        ----
            engine (``BaseEngine``): Specifies the engine.

        Returns:
        -------
            ``torch.nn.Module``, ``torch.optim.Optimizer``,
                ``Iterable``: A tuple with the model, the optimizer
                and the data loader.
        """
        logger.info("Preparing the model and optimizer...")
        model, optimizer = self._fabric.setup(engine.model, engine.optimizer, move_to_device=True)
        logger.info("Model and optimizer are ready")

        logger.info("Preparing the data loader...")
        data_loader = engine.data_source.get_data_loader(loader_id=self._tag, engine=engine)
        data_loader = self._fabric.setup_dataloaders(data_loader)
        prefix = f"({dist.get_rank()}/{dist.get_world_size()}) " if dist.is_distributed() else ""
        data_loader = tqdm(
            data_loader,
            desc=f"{prefix}Training [{engine.epoch}/{engine.max_epochs}]",
            position=dist.get_rank(),
            file=sys.stdout,
        )
        logger.info("Training data loader is ready")
        return model, optimizer, data_loader

    def _train_one_batch(
        self, engine: BaseEngine, model: Module, optimizer: Optimizer, batch: Any
    ) -> dict:
        engine.fire_event(EngineEvents.TRAIN_ITERATION_STARTED)
        optimizer.zero_grad(set_to_none=self._set_grad_to_none)
        output = model(batch)
        engine.fire_event(EngineEvents.TRAIN_FORWARD_COMPLETED)

        loss = output[ct.LOSS]
        if torch.isnan(loss):
            logger.warning(
                "NaN detected. The gradient is not computed for this batch "
                f"(iteration: {engine.iteration})"
            )
            engine.fire_event(EngineEvents.TRAIN_ITERATION_COMPLETED)
            return output

        self._fabric.backward(output[ct.LOSS])
        if self._clip_grad_fn:
            self._clip_grad_fn(model.parameters(), *self._clip_grad_args)
        engine.fire_event(EngineEvents.TRAIN_BACKWARD_COMPLETED)

        optimizer.step()
        engine.fire_event(EngineEvents.TRAIN_ITERATION_COMPLETED)

        return output

    def _setup_fabric(self, fabric: Union[Fabric, dict]) -> Fabric:
        r"""Sets up the ``lightning.Fabric`` object.

        Args:
        ----
            accelerator (``lightning.Fabric`` or dict, optional):
                Specifies the ``lightning.Fabric`` object or the
                parameters to instantiate it. Please read the
                ``lightning.Fabric`` documentation to know the
                parameters https://lightning.ai/docs/fabric/stable/.

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

    def _setup_clip_grad(self, config: dict) -> tuple[Optional[Callable], tuple]:
        return setup_clip_grad(config)
