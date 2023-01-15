r"""This module defines dome utility functions for the experiment trackers."""

__all__ = ["setup_exp_tracker", "main_process_only", "sanitize_metrics"]

import logging
from typing import Union

from gravitorch.distributed import comm as dist
from gravitorch.utils.exp_trackers.base import BaseExpTracker
from gravitorch.utils.exp_trackers.noop import NoOpExpTracker
from gravitorch.utils.format import str_target_object

logger = logging.getLogger(__name__)


def setup_exp_tracker(exp_tracker: Union[BaseExpTracker, dict, None]) -> BaseExpTracker:
    r"""Sets up the experiment tracker.

    Args:
        exp_tracker (``BaseExpTracker`` or dict or None): Specifies
            the experiment tracker or its configuration. if ``None``,
            the ``NoOpExpTracker`` is instantiated.

    Returns:
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


def main_process_only(exp_tracker: Union[BaseExpTracker, dict, None]) -> BaseExpTracker:
    r"""Instantiates the experiment tracker only for the main process.

    The non-main processes use the no-op experiment tracker.

    Args:
        exp_tracker (``BaseExpTracker`` or dict or None): Specifies
            the experiment tracker or its configuration. if ``None``,
            the ``NoOpExpTracker`` is instantiated.

    Returns:
        ``BaseExpTracker``: The instantiated experiment tracker
    """
    if dist.is_main_process():
        return setup_exp_tracker(exp_tracker)
    return NoOpExpTracker()


def sanitize_metrics(metrics: dict) -> dict[str, Union[int, float]]:
    r"""Sanitize a dictionary of metrics.

    This function removes all the values that are not an integer or
    float.

    Args:
        metrics (dict): Specifies the metrics to sanitize.

    Returns:
        dict: The sanitized metrics.
    """
    return {str(key): value for key, value in metrics.items() if isinstance(value, (int, float))}
