r"""This module implements a training loop using automatic mixed precision
(AMP)."""

__all__ = ["AMPTrainingLoop"]

import logging
from typing import Any, Optional, Union

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn import Module
from torch.optim import Optimizer

from gravitorch import constants as ct
from gravitorch.engines.base import BaseEngine
from gravitorch.engines.events import EngineEvents
from gravitorch.loops.observers import BaseLoopObserver
from gravitorch.loops.training.vanilla import VanillaTrainingLoop
from gravitorch.utils.device_placement import BaseDevicePlacement
from gravitorch.utils.profilers import BaseProfiler

logger = logging.getLogger(__name__)


class AMPTrainingLoop(VanillaTrainingLoop):
    r"""Implements a training loop to train a model on a dataset by using
    training loop using automatic mixed precision (AMP).

    Args:
        set_grad_to_none (bool, optional): If ``True``, set the
            gradients to ``None``, otherwise set the gradients to
            zero. Setting the gradients to ``None`` will in general
            have lower memory footprint, and can modestly improve
            performance. Default: ``False``
        amp_enabled (bool, optional): If ``True``, automatic mixed
            precision (AMP) is enabled, otherwise it is disabled.
            Default: ``True``
        batch_device_placement (bool, optional): Specifies the batch
            device placement module. This module moves the batch on
            a target device. The target device should be compatible
            with the model. If ``None``, an ``AutoDevicePlacement``
            object is instantiated. Default: ``None``
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
        clip_grad: Optional[dict] = None,
        set_grad_to_none: bool = False,
        amp_enabled: bool = True,
        batch_device_placement: Union[BaseDevicePlacement, dict, None] = None,
        tag: str = "train",
        observer: Union[BaseLoopObserver, dict, None] = None,
        profiler: Union[BaseProfiler, dict, None] = None,
    ):
        super().__init__(
            clip_grad=clip_grad,
            set_grad_to_none=set_grad_to_none,
            batch_device_placement=batch_device_placement,
            tag=tag,
            observer=observer,
            profiler=profiler,
        )
        self._amp_enabled = bool(amp_enabled)
        self._scaler = GradScaler(enabled=self._amp_enabled)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  set_grad_to_none={self._set_grad_to_none},\n"
            f"  amp_enabled={self._amp_enabled},\n"
            f"  tag={self._tag},\n"
            f"  batch_device_placement={self._batch_device_placement},\n"
            f"  clip_grad_fn={self._clip_grad_fn},\n"
            f"  clip_grad_args={self._clip_grad_args},\n"
            f"  observer={self._observer},\n"
            f"  profiler={self._profiler},\n"
            ")"
        )

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._scaler.load_state_dict(state_dict[ct.SCALER])

    def state_dict(self) -> dict[str, Any]:
        return {ct.SCALER: self._scaler.state_dict()}

    def _train_one_batch(
        self, engine: BaseEngine, model: Module, optimizer: Optimizer, batch: Any
    ) -> dict:
        engine.fire_event(EngineEvents.TRAIN_ITERATION_STARTED)
        optimizer.zero_grad(self._set_grad_to_none)
        with autocast(enabled=self._amp_enabled):
            output = model(self._batch_device_placement.send(batch))
        engine.fire_event(EngineEvents.TRAIN_FORWARD_COMPLETED)

        loss = self._scaler.scale(output[ct.LOSS])
        if torch.isnan(loss):
            logger.warning(
                "NaN detected. The gradient is not computed for this batch "
                f"(iteration: {engine.iteration})"
            )
            engine.fire_event(EngineEvents.TRAIN_ITERATION_COMPLETED)
            return output

        loss.backward()
        if self._clip_grad_fn:
            self._scaler.unscale_(optimizer)
            self._clip_grad_fn(model.parameters(), *self._clip_grad_args)
        engine.fire_event(EngineEvents.TRAIN_BACKWARD_COMPLETED)

        self._scaler.step(optimizer)
        self._scaler.update()
        engine.fire_event(EngineEvents.TRAIN_ITERATION_COMPLETED)

        return output
