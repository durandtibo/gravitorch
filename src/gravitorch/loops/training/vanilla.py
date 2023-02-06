r"""This module implements a simple training loop."""

__all__ = ["VanillaTrainingLoop"]

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
from gravitorch.utils.device_placement import (
    AutoDevicePlacement,
    BaseDevicePlacement,
    setup_device_placement,
)
from gravitorch.utils.profilers import BaseProfiler

logger = logging.getLogger(__name__)


class VanillaTrainingLoop(BaseBasicTrainingLoop):
    r"""Implements a simple training loop to train a model on a dataset.

    Args:
        set_grad_to_none (bool, optional): If ``True``, set the
            gradients to ``None``, otherwise set the gradients to
            zero. Setting the gradients to ``None`` will in general
            have lower memory footprint, and can modestly improve
            performance. Default: ``False``
        batch_device_placement (``BaseDevicePlacement`` or dict or
            ``None``, optional): Specifies the batch device placement
            module. This module moves the batch on a target device.
            The target device should be compatible with the model.
            If ``None``, an ``AutoDevicePlacement`` object is
            instantiated. Default: ``None``
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
        profiler (``BaseProfiler`` or dict or None, optional): Specifies
            the profiler or its configuration. If ``None``, the
            ``NoOpProfiler`` is instantiated. Default: ``None``
    """

    def __init__(
        self,
        set_grad_to_none: bool = False,
        batch_device_placement: Union[BaseDevicePlacement, dict, None] = None,
        tag: str = ct.TRAIN,
        clip_grad: Optional[dict] = None,
        observer: Union[BaseLoopObserver, dict, None] = None,
        profiler: Union[BaseProfiler, dict, None] = None,
    ):
        super().__init__(tag=tag, clip_grad=clip_grad, observer=observer, profiler=profiler)
        self._set_grad_to_none = bool(set_grad_to_none)
        self._batch_device_placement = setup_device_placement(
            batch_device_placement or AutoDevicePlacement()
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  tag={self._tag},\n"
            f"  set_grad_to_none={self._set_grad_to_none},\n"
            f"  batch_device_placement={self._batch_device_placement},\n"
            f"  clip_grad_fn={self._clip_grad_fn},\n"
            f"  clip_grad_args={self._clip_grad_args},\n"
            f"  observer={self._observer},\n"
            f"  profiler={self._profiler},\n"
            ")"
        )

    def _prepare_model_optimizer_data_loader(
        self, engine: BaseEngine
    ) -> tuple[Module, Optimizer, Iterable]:
        logger.info("Preparing the model, optimizer, and data loader...")
        data_loader = engine.data_source.get_data_loader(loader_id=self._tag, engine=engine)
        prefix = f"({dist.get_rank()}/{dist.get_world_size()}) " if dist.is_distributed() else ""
        data_loader = tqdm(
            data_loader,
            desc=f"{prefix}Training [{engine.epoch}/{engine.max_epochs}]",
            position=dist.get_rank(),
            file=sys.stdout,
        )
        logger.info("Training data loader has been created")
        return engine.model, engine.optimizer, data_loader

    def _train_one_batch(
        self, engine: BaseEngine, model: Module, optimizer: Optimizer, batch: Any
    ) -> dict:
        engine.fire_event(EngineEvents.TRAIN_ITERATION_STARTED)
        optimizer.zero_grad(self._set_grad_to_none)
        output = model(self._batch_device_placement.send(batch))
        engine.fire_event(EngineEvents.TRAIN_FORWARD_COMPLETED)

        loss = output[ct.LOSS]
        if torch.isnan(loss):
            logger.warning(
                "NaN detected. The gradient is not computed for this batch "
                f"(iteration: {engine.iteration})"
            )
            engine.fire_event(EngineEvents.TRAIN_ITERATION_COMPLETED)
            return output

        loss.backward()
        if self._clip_grad_fn:
            self._clip_grad_fn(model.parameters(), *self._clip_grad_args)
        engine.fire_event(EngineEvents.TRAIN_BACKWARD_COMPLETED)

        optimizer.step()
        engine.fire_event(EngineEvents.TRAIN_ITERATION_COMPLETED)

        return output

    def _setup_clip_grad(self, clip_grad: dict) -> tuple[Optional[Callable], tuple]:
        if not clip_grad:
            return None, ()

        name = clip_grad["name"]
        if name == "clip_grad_value":
            clip_grad_fn = torch.nn.utils.clip_grad_value_
            clip_grad_args = (clip_grad.get("clip_value", 0.25),)
            logger.info(f"clip gradient by value {clip_grad_args[0]}")
            return clip_grad_fn, clip_grad_args
        if name == "clip_grad_norm":
            clip_grad_fn = torch.nn.utils.clip_grad_norm_
            clip_grad_args = clip_grad.get("max_norm", 1), clip_grad.get("norm_type", 2)
            logger.info(
                f"clip gradient by maximum norm {clip_grad_args[0]} "
                f"(norm type: {clip_grad_args[1]})"
            )
            return clip_grad_fn, clip_grad_args
        raise ValueError(
            f"Incorrect clip grad name ({name}). The valid values are ``clip_grad_value`` "
            "and ``clip_grad_norm``"
        )
