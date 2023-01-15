__all__ = ["EvaluationRunner"]

import logging
from typing import Union

from gravitorch.distributed import comm as dist
from gravitorch.engines.base import BaseEngine
from gravitorch.handlers import setup_and_attach_handlers
from gravitorch.handlers.base import BaseHandler
from gravitorch.runners.distributed import BaseEngineDistributedRunner
from gravitorch.utils.cuda_memory import log_cuda_memory_summary
from gravitorch.utils.exp_trackers import BaseExpTracker, setup_exp_tracker
from gravitorch.utils.seed import manual_seed

logger = logging.getLogger(__name__)


class EvaluationRunner(BaseEngineDistributedRunner):
    r"""Implements a runner to evaluate a ML model.

    Internally, this runner does the following steps:

        - set the experiment tracker
        - set the random seed
        - instantiate the engine
        - set up and attach the handlers
        - evaluate the model with the engine
    """

    def _run(self) -> None:
        _run_evaluation_pipeline(
            engine=self._engine,
            handlers=self._handlers,
            exp_tracker=self._exp_tracker,
            random_seed=self._random_seed,
        )


def _run_evaluation_pipeline(
    engine: Union[BaseEngine, dict],
    handlers: Union[tuple[Union[BaseHandler, dict], ...], list[Union[BaseHandler, dict]]],
    exp_tracker: Union[BaseExpTracker, dict, None],
    random_seed: int = 8514665479832555083,
) -> BaseEngine:
    r"""Implements the evaluation pipeline.

    Internally, this function does the following steps:

        - set the experiment tracker
        - set the random seed
        - instantiate the engine
        - set up and attach the handlers
        - evaluate the model with the engine

    Args:
        engine (``BaseEngine`` or dict): Specifies the engine or its
            configuration.
        exp_tracker (``BaseExpTracker`` or dict or None):
            Specifies the experiment tracker or its configuration.
            If ``None``, the no-operation experiment tracker is used.
        random_seed (int, optional): Specifies the random seed.
            Default: ``8514665479832555083``

    Returns:
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
        engine.eval()

    log_cuda_memory_summary()
    return engine
