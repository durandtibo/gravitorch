r"""This module defines the base class for the training loops."""

__all__ = ["BaseBasicTrainingLoop"]

import logging
import sys
from abc import abstractmethod
from collections.abc import Callable, Iterable
from typing import Any, Optional, Union

from torch.nn import Module
from torch.optim import Optimizer

from gravitorch import constants as ct
from gravitorch.distributed import comm as dist
from gravitorch.engines.base import BaseEngine
from gravitorch.engines.events import EngineEvents
from gravitorch.loops.observers import BaseLoopObserver, setup_loop_observer
from gravitorch.loops.training.base import BaseTrainingLoop
from gravitorch.optimizers.utils import (
    log_optimizer_parameters_per_group,
    show_optimizer_parameters_per_group,
)
from gravitorch.utils.exp_trackers import EpochStep
from gravitorch.utils.history import MinScalarHistory
from gravitorch.utils.metric_tracker import ScalarMetricTracker
from gravitorch.utils.profilers import BaseProfiler, setup_profiler
from gravitorch.utils.seed import get_random_seed, manual_seed
from gravitorch.utils.timing import BatchLoadingTimer

logger = logging.getLogger(__name__)


class BaseBasicTrainingLoop(BaseTrainingLoop):
    r"""Implements a base class to implement basic training loops.

    Child classes have to implement the following methods:

        - ``_prepare_model_optimizer_data_loader``
        - ``_setup_clip_grad``
        - ``_train_one_batch``

    This class was implemented to reduce the duplicated code between
    ``VanillaTrainingLoop`` and ``AccelerateTrainingLoop``.

    Args:
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
            Specifies the profiler or its configuration.
            If ``None``, the ``NoOpProfiler`` is instantiated.
            Default: ``None``
    """

    def __init__(
        self,
        tag: str = ct.TRAIN,
        clip_grad: Optional[dict] = None,
        observer: Union[BaseLoopObserver, dict, None] = None,
        profiler: Union[BaseProfiler, dict, None] = None,
    ):
        self._tag = str(tag)
        self._clip_grad_fn, self._clip_grad_args = self._setup_clip_grad(clip_grad or {})
        self._observer = setup_loop_observer(observer)
        self._profiler = setup_profiler(profiler)
        logger.info(f"profiler:\n{self._profiler}")

    def train(self, engine: BaseEngine) -> None:
        dist.barrier()
        self._prepare_training(engine)
        engine.fire_event(EngineEvents.TRAIN_EPOCH_STARTED)

        model, optimizer, data_loader = self._prepare_model_optimizer_data_loader(engine)

        # Train the model on each mini-batch in the dataset.
        metrics = ScalarMetricTracker()
        data_loader = BatchLoadingTimer(data_loader, epoch=engine.epoch, prefix=f"{self._tag}/")
        self._observer.start(engine)
        dist.barrier()

        with self._profiler as profiler:
            for batch in data_loader:
                engine.increment_iteration()
                # Run forward/backward on the given batch.
                output = self._train_one_batch(engine, model, optimizer, batch)
                metrics.update(output)
                self._observer.update(engine=engine, model_input=batch, model_output=output)
                profiler.step()

        # To be sure the progress bar is displayed before the following lines
        sys.stdout.flush()
        dist.barrier()
        self._observer.end(engine)

        # Log some training metrics to the engine.
        data_loader.log_stats(engine=engine)
        metrics.log_average_value(engine=engine, prefix=f"{self._tag}/")
        dist.barrier()

        engine.fire_event(EngineEvents.TRAIN_EPOCH_COMPLETED)
        dist.barrier()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        pass

    def state_dict(self) -> dict[str, Any]:
        return {}

    def _prepare_training(self, engine: BaseEngine) -> None:
        r"""Prepares the training.

        Args:
            engine (``BaseEngine``): Specifies the engine.
        """
        logger.info(f"Preparing training for epoch {engine.epoch}...")
        manual_seed(
            get_random_seed(
                get_random_seed(
                    engine.random_seed + engine.epoch + engine.max_epochs * dist.get_rank()
                )
            )
        )
        engine.model.train()

        if not engine.has_history(f"{self._tag}/{ct.LOSS}"):
            engine.add_history(MinScalarHistory(f"{self._tag}/{ct.LOSS}"))

        show_optimizer_parameters_per_group(engine.optimizer)  # TODO: move to handler
        log_optimizer_parameters_per_group(
            optimizer=engine.optimizer,
            engine=engine,
            step=EpochStep(engine.epoch),
            prefix=f"{self._tag}/",
        )

    @abstractmethod
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

    @abstractmethod
    def _setup_clip_grad(self, clip_grad: dict) -> tuple[Optional[Callable], tuple]:
        r"""Initializes the clipping gradient strategy during training.

        Args:
            clip_grad (dict): Specifies the clipping gradient option.

        Returns:
            tuple: clip gradient function, clip gradient arguments.

        Raises:
            ValueError: if it is an invalid clipping gradient option.
        """

    @abstractmethod
    def _train_one_batch(
        self, engine: BaseEngine, model: Module, optimizer: Optimizer, batch: Any
    ) -> dict:
        """Trains the model on the given batch.

        Args:
            engine (``BaseEngine``): Specifies the engine.
            model (``torch.nn.Module``): Specifies the model to train.
            optimizer (``torch.optim.optimizer``): Specifies the
                optimizer used to train the model.
            batch: Specifies the batch of data.

        Returns:
            dict: Some results (including the loss value) about the
                batch.
        """
