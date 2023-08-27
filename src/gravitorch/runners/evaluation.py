from __future__ import annotations

__all__ = ["EvaluationRunner"]

import logging
from collections.abc import Sequence

from coola.utils import str_mapping

from gravitorch.distributed import comm as dist
from gravitorch.engines.base import BaseEngine
from gravitorch.handlers import setup_and_attach_handlers
from gravitorch.handlers.base import BaseHandler
from gravitorch.rsrc.base import BaseResource
from gravitorch.runners.resource import BaseResourceRunner
from gravitorch.utils.exp_trackers import BaseExpTracker, setup_exp_tracker
from gravitorch.utils.format import str_indent
from gravitorch.utils.seed import manual_seed

logger = logging.getLogger(__name__)


class EvaluationRunner(BaseResourceRunner):
    r"""Implements a runner to evaluate a ML model.

    Internally, this runner does the following steps:

        - set the experiment tracker
        - set the random seed
        - instantiate the engine
        - set up and attach the handlers
        - evaluate the model with the engine

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

        >>> from gravitorch.runners import EvaluationRunner
        >>> from gravitorch.testing import create_dummy_engine
        >>> engine = create_dummy_engine()
        >>> runner = EvaluationRunner(engine)
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
                    "handlers": self._handlers,
                    "resources": self._resources,
                    "random_seed": self._random_seed,
                }
            )
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def _run(self) -> BaseEngine:
        return _run_evaluation_pipeline(
            engine=self._engine,
            handlers=self._handlers,
            exp_tracker=self._exp_tracker,
            random_seed=self._random_seed,
        )


def _run_evaluation_pipeline(
    engine: BaseEngine | dict,
    handlers: tuple[BaseHandler | dict, ...] | list[BaseHandler | dict],
    exp_tracker: BaseExpTracker | dict | None,
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
    ----
        engine (``BaseEngine`` or dict): Specifies the engine or its
            configuration.
        exp_tracker (``BaseExpTracker`` or dict or None):
            Specifies the experiment tracker or its configuration.
            If ``None``, the no-operation experiment tracker is used.
        random_seed (int, optional): Specifies the random seed.
            Default: ``8514665479832555083``

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
        engine.eval()

    return engine
