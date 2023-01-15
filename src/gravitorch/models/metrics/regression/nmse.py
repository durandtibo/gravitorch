r"""This module defines the normalized mean squared error metric."""

__all__ = ["NormalizedMeanSquaredError"]

import logging
from typing import Optional

from torch import Tensor
from torch.nn import functional as F

from gravitorch.distributed.ddp import SUM, sync_reduce
from gravitorch.engines.base import BaseEngine
from gravitorch.models.metrics.base import EmptyMetricError
from gravitorch.models.metrics.base_epoch import BaseEpochMetric
from gravitorch.utils.exp_trackers import EpochStep
from gravitorch.utils.format import str_scalar
from gravitorch.utils.history import MinScalarHistory

logger = logging.getLogger(__name__)


class NormalizedMeanSquaredError(BaseEpochMetric):
    r"""Implements the normalized mean squared error (NMSE) metric.

    Note: this metric does not work if all the targets are zero.

    Args:
        mode (str): Specifies the mode (e.g. train or eval).
        name (str, optional): Specifies the name of the metric.
            The name is used to log the metric results.
            Default: ``'nmse'``
    """

    def __init__(self, mode: str, name: str = "nmse"):
        super().__init__(mode=mode, name=name)
        self._sum_squared_errors = 0.0
        self._sum_squared_targets = 0.0
        self._num_predictions = 0
        self.reset()

    def extra_repr(self) -> str:
        return f"mode={self._mode}, name={self._name}"

    def attach(self, engine: BaseEngine) -> None:
        r"""Attaches current metric to the provided engine.

        This method can be used to:

            - add event handler to the engine
            - set up history trackers

        Args:
            engine (``BaseEngine``): Specifies the engine.
        """
        super().attach(engine)
        engine.add_history(MinScalarHistory(name=self._metric_name))

    def forward(self, prediction: Tensor, target: Tensor) -> None:
        r"""Updates the mean squared error metric given a mini-batch of
        examples.

        Args:
            prediction (``torch.Tensor`` of shape
                ``(d0, d1, ..., dn)`` and type float or long):
                Specifies the tensor of predictions.
            target (``torch.Tensor`` of shape
                ``(d0, d1, ..., dn)`` and type float or long):
                Specifies the tensor of targets.
        """
        self._sum_squared_errors += F.mse_loss(
            prediction.float(), target.float(), reduction="sum"
        ).item()
        self._sum_squared_targets += target.pow(2).sum().item()
        self._num_predictions += target.numel()

    def reset(self) -> None:
        r"""Resets the metric."""
        self._sum_squared_errors = 0.0
        self._sum_squared_targets = 0.0
        self._num_predictions = 0

    def value(self, engine: Optional[BaseEngine] = None) -> dict:
        r"""Evaluates the metric and log the results given all the examples
        previously seen.

        Args:
            engine (``BaseEngine``, optional): Specifies the engine.
                This argument is required to log the results in the
                engine. Default: ``None``.

        Returns:
             dict: The results of the metric.
        """
        num_predictions = sync_reduce(self._num_predictions, op=SUM)
        if not num_predictions:
            raise EmptyMetricError(f"{self.__class__.__qualname__} is empty")

        results = {
            self._metric_name: sync_reduce(self._sum_squared_errors, op=SUM)
            / sync_reduce(self._sum_squared_targets, op=SUM),
            f"{self._metric_name}_num_predictions": num_predictions,
        }
        for name, value in results.items():
            logger.info(f"{name}: {str_scalar(value)}")
        if engine:
            engine.log_metrics(results, EpochStep(engine.epoch))
        return results
