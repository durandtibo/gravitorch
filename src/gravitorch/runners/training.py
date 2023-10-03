from __future__ import annotations

__all__ = ["TrainingRunner"]

import logging
from collections.abc import Sequence

from coola.utils import str_indent, str_mapping, str_sequence

from gravitorch.distributed import comm as dist
from gravitorch.engines.base import BaseEngine
from gravitorch.handlers import setup_and_attach_handlers
from gravitorch.handlers.base import BaseHandler
from gravitorch.rsrc.base import BaseResource
from gravitorch.runners.resource import BaseResourceRunner
from gravitorch.utils.exp_trackers import setup_exp_tracker
from gravitorch.utils.exp_trackers.base import BaseExpTracker
from gravitorch.utils.seed import manual_seed

logger = logging.getLogger(__name__)


class TrainingRunner(BaseResourceRunner):
    r"""Implements a runner to train a ML model.

    Internally, this runner does the following steps:

        - set the experiment tracker
        - set the random seed
        - instantiate the engine
        - set up and attach the handlers
        - train the model with the engine

    Args:
    ----
        engine (``BaseEngine`` or dict): Specifies the engine or its
            configuration.
        handlers (list or tuple or ``None``): Specifies the list of
            handlers or their configuration. The handlers will be
            attached to the engine. If ``None``, no handler is
            attached to the engine. Default: ``None``
        exp_tracker (``BaseExpTracker`` or dict or None): Specifies
            the experiment tracker or its configuration. If ``None``,
            the no-operation experiment tracker is used.
        random_seed (int, optional): Specifies the random seed.
            Default: ``10139531598155730726``
        resources (sequence or ``None``, optional): Specifies a
            sequence of resources or their configurations.
            Default: ``None``

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.runners import TrainingRunner
        >>> from gravitorch.testing import create_dummy_engine
        >>> engine = create_dummy_engine()
        >>> runner = TrainingRunner(engine)
        >>> runner  # doctest:+ELLIPSIS
        TrainingRunner(
          (engine): AlphaEngine(
              (datasource): DummyDataSource(
                  (datasets):
                    (train): DummyDataset(num_examples=4, feature_size=4)
                    (eval): DummyDataset(num_examples=4, feature_size=4)
                  (dataloader_creators):
                    (train): DataLoaderCreator(
                        (seed): 0
                        (batch_size): 2
                        (shuffle): False
                      )
                    (eval): DataLoaderCreator(
                        (seed): 0
                        (batch_size): 2
                        (shuffle): False
                      )
                )
              (model): DummyClassificationModel(
                  (linear): Linear(in_features=4, out_features=3, bias=True)
                  (criterion): CrossEntropyLoss()
                )
              (optimizer): SGD (
                Parameter Group 0
                    dampening: 0
                    differentiable: False
                    foreach: None
                    lr: 0.01
                    maximize: False
                    momentum: 0
                    nesterov: False
                    weight_decay: 0
                )
              (lr_scheduler): None
              (training_loop): TrainingLoop(
                  (set_grad_to_none): True
                  (batch_device_placement): AutoDevicePlacement(device=cpu)
                  (tag): train
                  (clip_grad_fn): None
                  (clip_grad_args): ()
                  (observer): NoOpLoopObserver()
                  (profiler): NoOpProfiler()
                )
              (evaluation_loop): EvaluationLoop(
                  (batch_device_placement): AutoDevicePlacement(device=cpu)
                  (grad_enabled): False
                  (tag): eval
                  (condition): EveryEpochEvalCondition(every=1)
                  (observer): NoOpLoopObserver()
                  (profiler): NoOpProfiler()
                )
              (state): EngineState(
                  (modules): AssetManager(num_assets=5)
                  (histories): HistoryManager()
                  (random_seed): 9984043075503325450
                  (max_epochs): 1
                  (epoch): -1
                  (iteration): -1
                )
              (event_manager): EventManager(
                  (event_handlers):
                  (last_triggered_event): None
                )
              (exp_tracker): NoOpExpTracker(experiment_path=..., is_activated=True))
          (exp_tracker): None
          (handlers): ()
          (resources): ()
          (random_seed): 10139531598155730726
        )
        >>> runner.run()
    """

    def __init__(
        self,
        engine: BaseEngine | dict,
        handlers: Sequence[BaseHandler | dict] | None = None,
        exp_tracker: BaseExpTracker | dict | None = None,
        random_seed: int = 10139531598155730726,
        resources: Sequence[BaseResource | dict] | None = None,
    ) -> None:
        super().__init__(resources=resources)
        self._engine = engine
        self._handlers = () if handlers is None else tuple(handlers)
        self._exp_tracker = exp_tracker
        self._random_seed = random_seed

    def __repr__(self) -> str:
        args = str_indent(
            str_mapping(
                {
                    "engine": self._engine,
                    "exp_tracker": self._exp_tracker,
                    "handlers": "\n" + str_sequence(self._handlers) if self._handlers else "()",
                    "resources": "\n" + str_sequence(self._resources) if self._resources else "()",
                    "random_seed": self._random_seed,
                }
            )
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def _run(self) -> BaseEngine:
        return _run_training_pipeline(
            engine=self._engine,
            handlers=self._handlers,
            exp_tracker=self._exp_tracker,
            random_seed=self._random_seed,
        )


def _run_training_pipeline(
    engine: BaseEngine | dict,
    handlers: tuple[BaseHandler | dict, ...] | list[BaseHandler | dict],
    exp_tracker: BaseExpTracker | dict | None,
    random_seed: int = 4398892194000378040,
) -> BaseEngine:
    r"""Implements the training pipeline.

    Internally, this function does the following steps:

        - set the experiment tracker
        - set the random seed
        - instantiate the engine
        - set up and attach the handlers
        - train the model with the engine

    Args:
    ----
        engine (``BaseEngine`` or dict): Specifies the engine or its
            configuration.
        handlers (list or tuple): Specifies the list of handlers or
            their configuration.
        exp_tracker (``BaseExpTracker`` or dict or None): Specifies
            the experiment tracker or its configuration. If ``None``,
            the no-operation experiment tracker is used.
        random_seed (int, optional): Specifies the random seed.
            Default: ``4398892194000378040``

    Returns:
    -------
        ``BaseEngine``: The trained engine.
    """
    with setup_exp_tracker(exp_tracker) as tracker:
        random_seed = random_seed + dist.get_rank()
        logger.info(f"Set the random seed to {random_seed}")
        manual_seed(random_seed)

        if isinstance(engine, dict):
            tracker.log_hyper_parameters(engine)
            logger.info("Initializing the engine from its configuration...")
            engine = BaseEngine.factory(exp_tracker=tracker, **engine)

        logger.info("Adding the handlers to the engine...")
        setup_and_attach_handlers(engine, handlers)

        logger.info(f"engine:\n{engine}")
        engine.train()

    return engine
