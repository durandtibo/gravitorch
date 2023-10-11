r"""This module implements a training loop using automatic mixed
precision (AMP)."""

from __future__ import annotations

__all__ = ["AMPTrainingLoop"]

import logging
from typing import Any

import torch
from coola.utils import str_indent, str_mapping
from torch.cuda.amp import GradScaler, autocast
from torch.nn import Module
from torch.optim import Optimizer

from gravitorch import constants as ct
from gravitorch.engines.base import BaseEngine
from gravitorch.engines.events import EngineEvents
from gravitorch.loops.observers import BaseLoopObserver
from gravitorch.loops.training.vanilla import TrainingLoop
from gravitorch.utils.device_placement import BaseDevicePlacement
from gravitorch.utils.profilers import BaseProfiler

logger = logging.getLogger(__name__)


class AMPTrainingLoop(TrainingLoop):
    r"""Implements a training loop to train a model on a dataset by using
    training loop using automatic mixed precision (AMP).

    Args:
    ----
        set_grad_to_none (bool, optional): If ``True``, set the
            gradients to ``None``, otherwise set the gradients to
            zero. Setting the gradients to ``None`` will in general
            have lower memory footprint, and can modestly improve
            performance. Default: ``True``
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

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.loops.training import AMPTrainingLoop
        >>> from gravitorch.testing import create_dummy_engine
        >>> engine = create_dummy_engine()
        >>> loop = AMPTrainingLoop()
        >>> loop
        AMPTrainingLoop(
          (set_grad_to_none): True
          (batch_device_placement): AutoDevicePlacement(device=cpu)
          (amp_enabled): True
          (tag): train
          (clip_grad_fn): None
          (clip_grad_args): ()
          (observer): NoOpLoopObserver()
          (profiler): NoOpProfiler()
        )
        >>> loop.train(engine)
    """

    def __init__(
        self,
        clip_grad: dict | None = None,
        set_grad_to_none: bool = True,
        amp_enabled: bool = True,
        batch_device_placement: BaseDevicePlacement | dict | None = None,
        tag: str = "train",
        observer: BaseLoopObserver | dict | None = None,
        profiler: BaseProfiler | dict | None = None,
    ) -> None:
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
        args = str_indent(
            str_mapping(
                {
                    "set_grad_to_none": self._set_grad_to_none,
                    "batch_device_placement": self._batch_device_placement,
                    "amp_enabled": self._amp_enabled,
                    "tag": self._tag,
                    "clip_grad_fn": self._clip_grad_fn,
                    "clip_grad_args": self._clip_grad_args,
                    "observer": self._observer,
                    "profiler": self._profiler,
                }
            )
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._scaler.load_state_dict(state_dict[ct.SCALER])

    def state_dict(self) -> dict[str, Any]:
        return {ct.SCALER: self._scaler.state_dict()}

    def _train_one_batch(
        self, engine: BaseEngine, model: Module, optimizer: Optimizer, batch: Any
    ) -> dict:
        engine.trigger_event(EngineEvents.TRAIN_ITERATION_STARTED)
        optimizer.zero_grad(self._set_grad_to_none)
        with autocast(enabled=self._amp_enabled):
            output = model(self._batch_device_placement.send(batch))
        engine.trigger_event(EngineEvents.TRAIN_FORWARD_COMPLETED)

        loss = self._scaler.scale(output[ct.LOSS])
        if torch.isnan(loss):
            logger.warning(
                "NaN detected in loss so backpropagation is skipped "
                f"(iteration: {engine.iteration})"
            )
            engine.trigger_event(EngineEvents.TRAIN_ITERATION_COMPLETED)
            return output

        loss.backward()
        if self._clip_grad_fn:
            self._scaler.unscale_(optimizer)
            self._clip_grad_fn(model.parameters(), *self._clip_grad_args)
        engine.trigger_event(EngineEvents.TRAIN_BACKWARD_COMPLETED)

        self._scaler.step(optimizer)
        self._scaler.update()
        engine.trigger_event(EngineEvents.TRAIN_ITERATION_COMPLETED)

        return output
