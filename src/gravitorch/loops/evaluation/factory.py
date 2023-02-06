r"""This module implements some utility functions for the evaluation loops."""

__all__ = ["setup_evaluation_loop"]

import logging
from typing import Union

from gravitorch.loops.evaluation.base import BaseEvaluationLoop
from gravitorch.loops.evaluation.vanilla import VanillaEvaluationLoop
from gravitorch.utils.format import str_target_object

logger = logging.getLogger(__name__)


def setup_evaluation_loop(
    evaluation_loop: Union[BaseEvaluationLoop, dict, None]
) -> BaseEvaluationLoop:
    r"""Sets up the evaluation loop.

    The evaluation loop is instantiated from its configuration by
    using the ``BaseEvaluationLoop`` factory function.

    Args:
        evaluation_loop (``BaseEvaluationLoop`` or dict or None):
            Specifies the evaluation loop or its configuration.
            If ``None``, the ``VanillaEvaluationLoop`` is instantiated.

    Returns:
        ``BaseEvaluationLoop``: The evaluation loop.
    """
    if evaluation_loop is None:
        evaluation_loop = VanillaEvaluationLoop()
    if isinstance(evaluation_loop, dict):
        logger.info(
            f"Initializing an evaluation loop from its configuration... "
            f"{str_target_object(evaluation_loop)}"
        )
        evaluation_loop = BaseEvaluationLoop.factory(**evaluation_loop)
    return evaluation_loop
