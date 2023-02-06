r"""This module implements a training loop using the Accelerate library
(https://huggingface.co/docs/accelerate)."""

__all__ = ["AccelerateTrainingLoop"]

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
from gravitorch.utils.integrations import check_accelerate, is_accelerate_available
from gravitorch.utils.profilers import BaseProfiler

if is_accelerate_available():
    from accelerate import Accelerator
else:
    Accelerator = None  # pragma: no cover

logger = logging.getLogger(__name__)


class AccelerateTrainingLoop(BaseBasicTrainingLoop):
    r"""Implements a training loop that uses ``accelerate.Accelerator`` to train
    a model.

    Args:
        accelerator (``accelerate.Accelerator`` or dict or None,
            optional): Specifies the ``accelerate.Accelerator`` object
            or the parameters to instantiate it. Please read the
            ``accelerate.Accelerator`` documentation to know the
            parameters
            https://huggingface.co/docs/accelerate/accelerator.html.
            If ``None``, it will use the default parameters.
            Default: ``None``
        set_grad_to_none (bool, optional): If ``True``, set the
            gradients to ``None``, otherwise set the gradients to
            zero. Setting the gradients to ``None`` will in general
            have lower memory footprint, and can modestly improve
            performance. Default: ``False``
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
        accelerator: Union[Accelerator, dict, None] = None,
        set_grad_to_none: bool = False,
        tag: str = "train",
        clip_grad: Optional[dict] = None,
        observer: Union[BaseLoopObserver, dict, None] = None,
        profiler: Union[BaseProfiler, dict, None] = None,
    ):
        check_accelerate()
        self._accelerator = self._setup_accelerator(accelerator or {})
        logger.info(f"accelerator state:\n{self._accelerator.state}")
        super().__init__(tag=tag, clip_grad=clip_grad, observer=observer, profiler=profiler)
        self._set_grad_to_none = bool(set_grad_to_none)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  accelerator={self._accelerator},\n"
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
            engine (``BaseEngine``): Specifies the engine.

        Returns:
            ``torch.nn.Module``, ``torch.optim.Optimizer``,
                ``Iterable``: A tuple with the model, the optimizer
                and the data loader.
        """
        logger.info("Preparing the model, optimizer, and data loader...")
        model, optimizer, data_loader = self._accelerator.prepare(
            [
                engine.model,
                engine.optimizer,
                engine.data_source.get_data_loader(loader_id=self._tag, engine=engine),
            ],
        )
        prefix = f"({dist.get_rank()}/{dist.get_world_size()}) " if dist.is_distributed() else ""
        data_loader = tqdm(
            data_loader,
            desc=f"{prefix}Training [{engine.epoch}/{engine.max_epochs}]",
            position=dist.get_rank(),
            file=sys.stdout,
        )
        logger.info("Training data loader has been created")
        return model, optimizer, data_loader

    def _train_one_batch(
        self, engine: BaseEngine, model: Module, optimizer: Optimizer, batch: Any
    ) -> dict:
        engine.fire_event(EngineEvents.TRAIN_ITERATION_STARTED)
        optimizer.zero_grad(self._set_grad_to_none)
        output = model(batch)
        engine.fire_event(EngineEvents.TRAIN_FORWARD_COMPLETED)

        if not torch.isnan(output[ct.LOSS]):
            self._accelerator.backward(output[ct.LOSS])
        else:
            logger.warning(
                "NaN detected. The gradient is not computed for this batch "
                f"(iteration: {engine.iteration})"
            )

        if self._clip_grad_fn:
            self._clip_grad_fn(model.parameters(), *self._clip_grad_args)
        engine.fire_event(EngineEvents.TRAIN_BACKWARD_COMPLETED)

        optimizer.step()
        engine.fire_event(EngineEvents.TRAIN_ITERATION_COMPLETED)

        return output

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

    def _setup_clip_grad(self, clip_grad: dict) -> tuple[Optional[Callable], tuple]:
        if not clip_grad:
            return None, ()

        name = clip_grad["name"]
        if name == "clip_grad_value":
            clip_grad_fn = self._accelerator.clip_grad_value_
            clip_grad_args = (clip_grad.get("clip_value", 0.25),)
            logger.info(f"clip gradient by value {clip_grad_args[0]}")
            return clip_grad_fn, clip_grad_args
        if name == "clip_grad_norm":
            clip_grad_fn = self._accelerator.clip_grad_norm_
            clip_grad_args = clip_grad.get("max_norm", 1), clip_grad.get("norm_type", 2)
            logger.info(
                f"clip gradient by maximum norm {clip_grad_args[0]} (norm type: "
                f"{clip_grad_args[1]})"
            )
            return clip_grad_fn, clip_grad_args
        raise ValueError(
            f"Incorrect clip grad name ({name}). The valid values are ``clip_grad_value`` "
            "and ``clip_grad_norm``"
        )
