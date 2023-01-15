r"""This module implements some meters."""

from __future__ import annotations

__all__ = ["ScalarMetricTracker"]

import logging
import math
from collections import defaultdict
from typing import TYPE_CHECKING

import torch

from gravitorch.utils.exp_trackers.steps import EpochStep
from gravitorch.utils.format import str_scalar
from gravitorch.utils.meters.scalar import ScalarMeter

if TYPE_CHECKING:
    from gravitorch.engines import BaseEngine

logger = logging.getLogger(__name__)


class ScalarMetricTracker:
    r"""Implements a tracker for multiple scalar metrics.

    This tracker use the ``ScalarMeter`` to track some values of each
    metric.

    Example usage:

    .. code-block:: python

        >>> from gravitorch.utils.metric_tracker import ScalarMetricTracker
        >>> tracker = ScalarMetricTracker()
        >>> tracker.update({'metric1': 1, 'metric2': 2})
        >>> tracker.update({'metric1': 11, 'metric2': 12})
        >>> tracker.update({'metric1': 21, 'metric2': 22})
        >>> tracker.log_average_value()
        INFO:gravitorch.utils.metric_tracker:metric1: 11.0000
        INFO:gravitorch.utils.metric_tracker:metric2: 12.0000
    """

    def __init__(self):
        self._metrics = defaultdict(ScalarMeter)

    def update(self, data: dict) -> None:
        r"""Tracks the values that can be convert in int or float.

        The values that cannot be convert in int or float are ignored.
        Similarly, the NaN are ignored.

        Args:
            data (dict): Specifies a dict with the metric values to
                track.

        Example usage:

        .. code-block:: python

            >>> from gravitorch.utils.metric_tracker import ScalarMetricTracker
            >>> tracker = ScalarMetricTracker()
            >>> tracker.update({'metric1': 1, 'metric2': 2})
        """
        for key, value in data.items():
            if torch.is_tensor(value) and value.numel() == 1:
                value = value.item()  # get number from a torch scalar
            if isinstance(value, (int, float)) and not math.isnan(value):
                self._metrics[key].update(value)

    def log_average_value(self, engine: BaseEngine | None = None, prefix: str = "") -> None:
        r"""Logs the average value of the metrics to the engine.

        Args:
            engine (``BaseEngine`` or None, optional): Specifies the
                engine. Default: ``None``
            prefix (str, optional): Specifies the prefix used to log
                the metric name. Default: ``''``

        Example usage:

        .. code-block:: python

            >>> from gravitorch.utils.metric_tracker import ScalarMetricTracker
            >>> tracker = ScalarMetricTracker()
            >>> tracker.update({'metric1': 1, 'metric2': 2})
            >>> tracker.log_average_value()
            INFO:gravitorch.utils.metric_tracker:metric1: 1.000000
            INFO:gravitorch.utils.metric_tracker:metric2: 2.000000
            >>> tracker.log_average_value(prefix='train/')
            INFO:gravitorch.utils.metric_tracker:train/metric1: 1.000000
            INFO:gravitorch.utils.metric_tracker:train/metric2: 2.000000
        """
        metrics = {f"{prefix}{key}": value.average() for key, value in self._metrics.items()}
        if engine:
            engine.log_metrics(metrics, step=EpochStep(engine.epoch))
        for name, value in metrics.items():
            logger.info(f"{name}: {str_scalar(value)}")
