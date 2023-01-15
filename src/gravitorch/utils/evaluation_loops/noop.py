r"""This module implements a no-operation evaluation loop."""

__all__ = ["NoOpEvaluationLoop"]

import logging

from gravitorch.engines.base import BaseEngine
from gravitorch.utils.evaluation_loops.base import BaseEvaluationLoop

logger = logging.getLogger(__name__)


class NoOpEvaluationLoop(BaseEvaluationLoop):
    r"""Implements a no-operation evaluation loop.

    This class can be used to ignore the evaluation loop in an engine.
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def eval(self, engine: BaseEngine) -> None:
        r"""Evaluates the model on the evaluation dataset.

        It is a no-operation method.

        Args:
            engine (``BaseEngine``): Specifies the engine.
        """
