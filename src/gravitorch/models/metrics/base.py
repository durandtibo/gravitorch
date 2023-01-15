r"""Base module for metrics."""

__all__ = ["BaseMetric", "EmptyMetricError", "setup_metric"]

import logging
from abc import abstractmethod
from typing import Optional, Union

from objectory import AbstractFactory
from torch.nn import Module

from gravitorch.engines.base import BaseEngine
from gravitorch.utils.format import str_target_object

logger = logging.getLogger(__name__)


class BaseMetric(Module, metaclass=AbstractFactory):
    r"""Defines the base class for the metric.

    This class is used to register the metric using the metaclass
    factory. Child classes must implement the following methods:
        - ``attach``
        - ``forward``
        - ``reset``
        - ``value``
    """

    @abstractmethod
    def attach(self, engine: BaseEngine) -> None:
        r"""Attaches current metric to provided engine.

        This method can be used to:

            - add event handler to the engine
            - set up history trackers

        Args:
            engine (``BaseEngine``): Specifies the engine.
        """

    @abstractmethod
    def forward(self, *args, **kwargs) -> Optional[dict]:
        r"""Updates the metric given a mini-batch of examples.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """

    @abstractmethod
    def reset(self) -> None:
        r"""Resets the metric."""

    @abstractmethod
    def value(self, engine: Optional[BaseEngine] = None) -> dict:
        r"""Evaluates the metric and log the results given all the examples
        previously seen.

        Args:
            engine (``BaseEngine`` or None): Specifies the engine.

        Returns:
             dict: The results of the metric.
        """


class EmptyMetricError(Exception):
    r"""This error is raised when you try to evaluate an empty metric."""


def setup_metric(metric: Union[BaseMetric, dict]) -> BaseMetric:
    r"""Sets up the metric.

    Args:
        metric (``BaseMetric`` or dict): Specifies the metric or its
            configuration.

    Returns:
        ``BaseMetric``: The instantiated metric.
    """
    if isinstance(metric, dict):
        logger.info(f"Initializing a metric from its configuration... {str_target_object(metric)}")
        metric = BaseMetric.factory(**metric)
    return metric
