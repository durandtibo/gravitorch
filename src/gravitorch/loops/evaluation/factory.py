r"""This module implements some utility functions for the evaluation
loops."""

from __future__ import annotations

__all__ = ["is_evaluation_loop_config", "setup_evaluation_loop"]

import logging

from objectory.utils import is_object_config

from gravitorch.loops.evaluation.base import BaseEvaluationLoop
from gravitorch.loops.evaluation.vanilla import VanillaEvaluationLoop
from gravitorch.utils.format import str_target_object

logger = logging.getLogger(__name__)


def is_evaluation_loop_config(config: dict) -> bool:
    r"""Indicate if the input configuration is a configuration for a
    ``BaseEvaluationLoop``.

    This function only checks if the value of the key  ``_target_``
    is valid. It does not check the other values. If ``_target_``
    indicates a function, the returned type hint is used to check
    the class.

    Args:
    ----
        config (dict): Specifies the configuration to check.

    Returns:
    -------
        bool: ``True`` if the input configuration is a configuration
            for a ``BaseEvaluationLoop`` object.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.loops.evaluation import is_evaluation_loop_config
        >>> is_evaluation_loop_config(
        ...     {"_target_": "gravitorch.loops.evaluation.VanillaEvaluationLoop"}
        ... )
        True
    """
    return is_object_config(config, BaseEvaluationLoop)


def setup_evaluation_loop(evaluation_loop: BaseEvaluationLoop | dict | None) -> BaseEvaluationLoop:
    r"""Sets up the evaluation loop.

    The evaluation loop is instantiated from its configuration by
    using the ``BaseEvaluationLoop`` factory function.

    Args:
    ----
        evaluation_loop (``BaseEvaluationLoop`` or dict or None):
            Specifies the evaluation loop or its configuration.
            If ``None``, the ``VanillaEvaluationLoop`` is instantiated.

    Returns:
    -------
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
    if not isinstance(evaluation_loop, BaseEvaluationLoop):
        logger.warning(
            f"evaluation_loop is not a `BaseEvaluationLoop` (received: {type(evaluation_loop)})"
        )
    return evaluation_loop
