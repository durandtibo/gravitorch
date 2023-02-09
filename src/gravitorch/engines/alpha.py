from __future__ import annotations

__all__ = ["AlphaEngine"]

import logging
from typing import TYPE_CHECKING, Any

from torch.nn import Module
from torch.optim import Optimizer

from gravitorch import constants as ct
from gravitorch.creators.core.base import BaseCoreCreator, setup_core_creator
from gravitorch.distributed import comm as dist
from gravitorch.engines.base import BaseEngine
from gravitorch.engines.events import EngineEvents
from gravitorch.loops.evaluation import BaseEvaluationLoop, setup_evaluation_loop
from gravitorch.loops.training import BaseTrainingLoop, setup_training_loop
from gravitorch.lr_schedulers import LRSchedulerType
from gravitorch.utils.artifacts import BaseArtifact
from gravitorch.utils.engine_states import BaseEngineState, setup_engine_state
from gravitorch.utils.events import BaseEventHandler, EventManager
from gravitorch.utils.exp_trackers import BaseExpTracker, Step, setup_exp_tracker
from gravitorch.utils.format import str_indent
from gravitorch.utils.history import BaseHistory
from gravitorch.utils.timing import timeblock

if TYPE_CHECKING:
    from gravitorch.datasources import BaseDataSource

logger = logging.getLogger(__name__)


class AlphaEngine(BaseEngine):
    r"""Implements an engine to train or evaluate a model.

    The core engine modules of this engine are created by a core
    engine modules creator.

    Args:
        core_creator (``BaseCoreCreator`` or dict):
            Specifies the core engine modules creators or its
            configuration. This object creates the data module, model,
            optimizer and LR scheduler.
        state (``BaseEngineState`` or dict or ``None``): Specifies
            the engine state or its configuration. If ``None``, a
            ``VanillaEngineState`` object is created. The state is
            used to represent all the modules added to the engine,
            but the event system is not include in the engine state.
            Default: ``None``
        exp_tracker (``BaseExpTracker`` or dict or ``None``):
            Specifies the experiment tracker or its configuration.
            If ``None``, a ``NoOpExpTracker`` object is created.
            The experiment tracker is used to track some artifacts
            in a third-party library.
            Default: ``None``
        training_loop (``BaseTrainingLoop`` or dict or ``None``):
            Specifies the training loop or its configuration.
            If ``None``, a ``VanillaTrainingLoop`` object is created.
            Default: ``None``
        evaluation_loop (``BaseEvaluationLoop`` or dict or ``None``):
            Specifies the evaluation loop or its configuration.
            If ``None``, a ``VanillaEvaluationLoop`` object is
            created. Default: ``None``
    """

    def __init__(
        self,
        core_creator: BaseCoreCreator | dict,
        state: BaseEngineState | dict | None = None,
        exp_tracker: BaseExpTracker | dict | None = None,
        training_loop: BaseTrainingLoop | dict | None = None,
        evaluation_loop: BaseEvaluationLoop | dict | None = None,
    ):
        self._event_manager = EventManager()
        self._state = self._setup_state(state)
        self._exp_tracker = self._setup_exp_tracker(exp_tracker)
        self._training_loop = self._setup_training_loop(training_loop)
        self._evaluation_loop = self._setup_evaluation_loop(evaluation_loop)

        core_creator = setup_core_creator(core_creator)
        logger.info(f"core_creator:\n{core_creator}")
        (
            self._data_source,
            self._model,
            self._optimizer,
            self._lr_scheduler,
        ) = core_creator.create(self)
        # TODO: add the modules if there are not added yet.

        self._should_terminate = False

    def __str__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  data_source={str_indent(self._data_source)},\n"
            f"  model={str_indent(self._model)},\n"
            f"  optimizer={str_indent(self._optimizer)},\n"
            f"  lr_scheduler={str_indent(self._lr_scheduler)},\n"
            f"  training_loop={str_indent(self._training_loop)},\n"
            f"  evaluation_loop={str_indent(self._evaluation_loop)},\n"
            f"  state={str_indent(self._state)},\n"
            f"  event_manager={str_indent(self._event_manager)},\n"
            f"  exp_tracker={str_indent(self._exp_tracker)},\n"
            ")"
        )

    @property
    def data_source(self) -> BaseDataSource:
        return self._data_source

    @property
    def epoch(self) -> int:
        return self._state.epoch

    @property
    def iteration(self) -> int:
        return self._state.iteration

    @property
    def lr_scheduler(self) -> LRSchedulerType | None:
        return self._lr_scheduler

    @property
    def max_epochs(self) -> int:
        return self._state.max_epochs

    @property
    def model(self) -> Module:
        return self._model

    @property
    def optimizer(self) -> Optimizer | None:
        return self._optimizer

    @property
    def random_seed(self) -> int:
        return self._state.random_seed

    @property
    def should_terminate(self) -> bool:
        return self._should_terminate

    def add_event_handler(self, event: str, event_handler: BaseEventHandler) -> None:
        self._event_manager.add_event_handler(event, event_handler)

    def add_history(self, history: BaseHistory, key: str | None = None) -> None:
        self._state.add_history(history=history, key=key)

    def add_module(self, name: str, module: Any) -> None:
        self._state.add_module(name=name, module=module)

    def create_artifact(self, artifact: BaseArtifact) -> None:
        self._exp_tracker.create_artifact(artifact)

    def eval(self) -> None:
        with timeblock("=== Evaluation time: {time} ==="):
            logger.info("Launching evaluation procedures")
            dist.barrier()
            self.fire_event(EngineEvents.STARTED)
            self._evaluation_loop.eval(self)
            dist.barrier()
            self.fire_event(EngineEvents.COMPLETED)
            self._exp_tracker.flush()
            logger.info("Ending evaluation procedures")

    def fire_event(self, event: str) -> None:
        self._event_manager.fire_event(event)

    def get_history(self, key: str) -> BaseHistory:
        return self._state.get_history(key)

    def get_histories(self) -> dict[str, BaseHistory]:
        return self._state.get_histories()

    def get_module(self, name: str) -> Any:
        return self._state.get_module(name)

    def has_event_handler(self, event_handler: BaseEventHandler, event: str | None = None) -> bool:
        return self._event_manager.has_event_handler(event_handler, event)

    def has_history(self, key: str) -> bool:
        return self._state.has_history(key)

    def has_module(self, name: str) -> bool:
        return self._state.has_module(name)

    def increment_epoch(self, increment: int = 1) -> None:
        self._state.increment_epoch(increment)

    def increment_iteration(self, increment: int = 1) -> None:
        self._state.increment_iteration(increment)

    def load_state_dict(self, state_dict: dict) -> None:
        self._state.load_state_dict(state_dict)

    def log_figure(
        self,
        key: str,
        figure: matplotlib.pyplot.Figure,  # noqa: F821
        step: Step | None = None,
    ) -> None:
        self._exp_tracker.log_figure(key, figure, step)

    def log_figures(
        self,
        figures: dict[str, matplotlib.pyplot.Figure],  # noqa: F821
        step: Step | None = None,
    ) -> None:
        self._exp_tracker.log_figures(figures, step)

    def log_metric(self, key: str, value: int | float, step: Step | None = None):
        self._state.get_history(key).add_value(value, step.step if step else step)
        self._exp_tracker.log_metric(key, value, step)

    def log_metrics(self, metrics: dict[str, int | float], step: Step | None = None) -> None:
        for key, value in metrics.items():
            self._state.get_history(key).add_value(value, step.step if step else step)
        self._exp_tracker.log_metrics(metrics, step)

    def remove_event_handler(self, event: str, event_handler: BaseEventHandler) -> None:
        self._event_manager.remove_event_handler(event, event_handler)

    def remove_module(self, name: str) -> None:
        self._state.remove_module(name)

    def state_dict(self) -> dict[str, Any]:
        return self._state.state_dict()

    def terminate(self) -> None:
        self._should_terminate = True

    def train(self) -> None:
        with timeblock("=== Training time: {time} ==="):
            logger.info("Launching training procedures")
            self.fire_event(EngineEvents.STARTED)
            self.fire_event(EngineEvents.TRAIN_STARTED)
            if not self._should_terminate:
                for _ in range(self.epoch + 1, self.max_epochs):
                    self.increment_epoch()
                    dist.barrier()
                    self.fire_event(EngineEvents.EPOCH_STARTED)

                    self._training_loop.train(self)
                    self._evaluation_loop.eval(self)

                    dist.barrier()
                    self.fire_event(EngineEvents.EPOCH_COMPLETED)
                    self._log_best_metrics()
                    self._exp_tracker.flush()

                    if self._should_terminate:
                        break

                dist.barrier()
                self.fire_event(EngineEvents.TRAIN_COMPLETED)

            self.fire_event(EngineEvents.COMPLETED)
            dist.barrier()
            self._exp_tracker.flush()
            logger.info("Ending training procedures...")

    def _log_best_metrics(self) -> None:
        r"""Logs the best value of each scalar comparable metric."""
        dist.barrier()
        best_metrics = self._state.get_best_values()
        best_metrics = {
            key: value for key, value in best_metrics.items() if isinstance(value, (int, float))
        }
        self._exp_tracker.log_best_metrics(best_metrics)

    def _setup_evaluation_loop(
        self, evaluation_loop: BaseEvaluationLoop | dict | None
    ) -> BaseEvaluationLoop:
        r"""Sets up the evaluation loop.

        The evaluation loop is instantiated from its configuration
        by using the ``BaseEvaluationLoop`` factory function.

        Args:
            evaluation_loop (``BaseEvaluationLoop`` or dict or None):
                Specifies the evaluation loop or its configuration.
                If ``None``, the ``VanillaEvaluationLoop`` is
                instantiated.

        Returns:
            ``BaseEvaluationLoop``: The evaluation loop.
        """
        evaluation_loop = setup_evaluation_loop(evaluation_loop)
        logger.info(f"evaluation loop:\n{evaluation_loop}")
        self.add_module(ct.EVALUATION_LOOP, evaluation_loop)
        return evaluation_loop

    def _setup_exp_tracker(self, exp_tracker: BaseExpTracker | dict | None) -> BaseExpTracker:
        r"""Sets up the experiment tracker.

        If the input is None, the no-operation experiment tracker is
        used. The experiment tracker is instantiated from its
        configuration by using the ``BaseExpTracker`` factory function.

        Args:
            exp_tracker (``BaseExpTracker`` or dict or None):
                Specifies the experiment tracker or its configuration.

        Returns:
            ``BaseExpTracker``: The (instantiated) experiment tracker.
        """
        exp_tracker = setup_exp_tracker(exp_tracker)
        logger.info(f"experiment tracker:\n{exp_tracker}")
        if not exp_tracker.is_activated():
            exp_tracker.start()
        return exp_tracker

    def _setup_state(self, state: BaseEngineState | dict | None) -> BaseEngineState:
        r"""Sets up the engine state.

        The engine state is instantiated from its configuration by
        using the ``BaseEngineState`` factory function.

        Args:
            state (``BaseEngineState`` or dict or None):
                Specifies the engine state or its configuration.
                If ``None``, the ``VanillaEngineState`` is
                instantiated.

        Returns:
            ``BaseEngineState``: The engine state.
        """
        state = setup_engine_state(state)
        logger.info(f"state:\n{state}")
        return state

    def _setup_training_loop(
        self, training_loop: BaseTrainingLoop | dict | None
    ) -> BaseTrainingLoop:
        r"""Sets up the training loop.

        The training loop is instantiated from its configuration by
        using the ``BaseTrainingLoop`` factory function.

        Args:
            training_loop (``BaseTrainingLoop`` or dict or None):
                Specifies the training loop or its configuration.
                If ``None``, the ``VanillaTrainingLoop`` is
                instantiated.

        Returns:
            ``BaseTrainingLoop``: The training loop.
        """
        training_loop = setup_training_loop(training_loop)
        logger.info(f"training loop:\n{training_loop}")
        self.add_module(ct.TRAINING_LOOP, training_loop)
        return training_loop
