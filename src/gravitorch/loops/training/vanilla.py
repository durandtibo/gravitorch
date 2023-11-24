r"""This module implements a simple training loop."""

from __future__ import annotations

__all__ = ["TrainingLoop"]

import logging
from collections.abc import Callable, Iterable
from typing import Any

import torch
from coola.utils import str_indent, str_mapping
from torch.nn import Module
from torch.optim import Optimizer

from gravitorch import constants as ct
from gravitorch.engines.base import BaseEngine
from gravitorch.engines.events import EngineEvents
from gravitorch.loops.observers import BaseLoopObserver
from gravitorch.loops.training.basic import BaseBasicTrainingLoop
from gravitorch.loops.training.utils import setup_clip_grad
from gravitorch.utils.device_placement import (
    AutoDevicePlacement,
    BaseDevicePlacement,
    setup_device_placement,
)
from gravitorch.utils.profilers import BaseProfiler

logger = logging.getLogger(__name__)


class TrainingLoop(BaseBasicTrainingLoop):
    r"""Implements a simple training loop to train a model on a dataset.

    Args:
    ----
        set_grad_to_none (bool, optional): If ``True``, set the
            gradients to ``None``, otherwise set the gradients to
            zero. Setting the gradients to ``None`` will in general
            have lower memory footprint, and can modestly improve
            performance. Default: ``True``
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

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.loops.training import TrainingLoop
        >>> from gravitorch.testing import create_dummy_engine
        >>> engine = create_dummy_engine()
        >>> loop = TrainingLoop()
        >>> loop
        TrainingLoop(
          (set_grad_to_none): True
          (batch_device_placement): AutoDevicePlacement(device=cpu)
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
        set_grad_to_none: bool = True,
        batch_device_placement: BaseDevicePlacement | dict | None = None,
        tag: str = ct.TRAIN,
        clip_grad: dict | None = None,
        observer: BaseLoopObserver | dict | None = None,
        profiler: BaseProfiler | dict | None = None,
    ) -> None:
        super().__init__(tag=tag, clip_grad=clip_grad, observer=observer, profiler=profiler)
        self._set_grad_to_none = bool(set_grad_to_none)
        self._batch_device_placement = setup_device_placement(
            batch_device_placement or AutoDevicePlacement()
        )

    def __repr__(self) -> str:
        args = str_indent(
            str_mapping(
                {
                    "set_grad_to_none": self._set_grad_to_none,
                    "batch_device_placement": self._batch_device_placement,
                    "tag": self._tag,
                    "clip_grad_fn": self._clip_grad_fn,
                    "clip_grad_args": self._clip_grad_args,
                    "observer": self._observer,
                    "profiler": self._profiler,
                }
            )
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def _prepare_model_optimizer_dataiter(
        self, engine: BaseEngine
    ) -> tuple[Module, Optimizer, Iterable]:
        logger.info("Preparing the model, optimizer, and data iterable...")
        dataiter = engine.datasource.get_iterable(iter_id=self._tag, engine=engine)
        logger.info("Training model, optimizer, and data iterable have been created")
        return engine.model, engine.optimizer, dataiter

    def _train_one_batch(
        self, engine: BaseEngine, model: Module, optimizer: Optimizer, batch: Any
    ) -> dict:
        engine.trigger_event(EngineEvents.TRAIN_ITERATION_STARTED)
        optimizer.zero_grad(self._set_grad_to_none)
        output = model(self._batch_device_placement.send(batch))
        engine.trigger_event(EngineEvents.TRAIN_FORWARD_COMPLETED)

        loss = output[ct.LOSS]
        if torch.isnan(loss):
            logger.warning(
                "NaN detected in loss so backpropagation is skipped "
                f"(iteration: {engine.iteration})"
            )
            engine.trigger_event(EngineEvents.TRAIN_ITERATION_COMPLETED)
            return output

        loss.backward()
        if self._clip_grad_fn:
            self._clip_grad_fn(model.parameters(), *self._clip_grad_args)
        engine.trigger_event(EngineEvents.TRAIN_BACKWARD_COMPLETED)

        optimizer.step()
        engine.trigger_event(EngineEvents.TRAIN_ITERATION_COMPLETED)

        return output

    def _setup_clip_grad(self, config: dict) -> tuple[Callable | None, tuple]:
        return setup_clip_grad(config)
