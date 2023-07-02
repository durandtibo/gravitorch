r"""This module defines dome utility functions for the experiment
trackers."""
from __future__ import annotations

__all__ = [
    "setup_exp_tracker",
    "main_process_only",
    "sanitize_metrics",
    "is_exp_tracker_config",
]

import logging

from objectory.utils import is_object_config

from gravitorch.distributed import comm as dist
from gravitorch.utils.exp_trackers.base import BaseExpTracker
from gravitorch.utils.exp_trackers.noop import NoOpExpTracker
from gravitorch.utils.format import str_target_object

logger = logging.getLogger(__name__)


def is_exp_tracker_config(config: dict) -> bool:
    r"""Indicate if the input configuration is a configuration for a
    ``BaseHandler``.

    This function only checks if the value of the key  ``_target_``
    is valid. It does not check the other values.

    Args:
    ----
        config (dict): Specifies the configuration to check.

    Returns:
    -------
        bool: ``True`` if the input configuration is a configuration
            for a ``BaseHandler`` object.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.utils.exp_trackers import is_exp_tracker_config
        >>> is_exp_tracker_config({"_target_": "gravitorch.utils.exp_trackers.NoOpExpTracker"})
        True
    """
    return is_object_config(config, BaseExpTracker)


def setup_exp_tracker(exp_tracker: BaseExpTracker | dict | None) -> BaseExpTracker:
    r"""Sets up the experiment tracker.

    Args:
    ----
        exp_tracker (``BaseExpTracker`` or dict or None): Specifies
            the experiment tracker or its configuration. if ``None``,
            the ``NoOpExpTracker`` is instantiated.

    Returns:
    -------
        ``BaseExpTracker``: The experiment tracker
    """
    if exp_tracker is None:
        exp_tracker = NoOpExpTracker()
    if isinstance(exp_tracker, dict):
        logger.info(
            "Initializing an experiment tracker from its configuration... "
            f"{str_target_object(exp_tracker)}"
        )
        exp_tracker = BaseExpTracker.factory(**exp_tracker)
    return exp_tracker


def main_process_only(exp_tracker: BaseExpTracker | dict | None) -> BaseExpTracker:
    r"""Instantiates the experiment tracker only for the main process.

    The non-main processes use the no-op experiment tracker.

    Args:
    ----
        exp_tracker (``BaseExpTracker`` or dict or None): Specifies
            the experiment tracker or its configuration. if ``None``,
            the ``NoOpExpTracker`` is instantiated.

    Returns:
    -------
        ``BaseExpTracker``: The instantiated experiment tracker
    """
    if dist.is_main_process():
        return setup_exp_tracker(exp_tracker)
    return NoOpExpTracker()


def sanitize_metrics(metrics: dict) -> dict[str, int | float]:
    r"""Sanitize a dictionary of metrics.

    This function removes all the values that are not an integer or
    float.

    Args:
    ----
        metrics (dict): Specifies the metrics to sanitize.

    Returns:
    -------
        dict: The sanitized metrics.
    """
    return {str(key): value for key, value in metrics.items() if isinstance(value, (int, float))}
