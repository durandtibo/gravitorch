r"""This module defines the normalized mean squared error metric."""

from __future__ import annotations

__all__ = ["NormalizedMeanSquaredError"]

import logging

from torch import Tensor
from torch.nn.functional import mse_loss

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
    ----
        mode (str): Specifies the mode (e.g. train or eval).
        name (str, optional): Specifies the name of the metric.
            The name is used to log the metric results.
            Default: ``'nmse'``

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from gravitorch.models.metrics import NormalizedMeanSquaredError
        >>> metric = NormalizedMeanSquaredError("eval")
        >>> metric
        NormalizedMeanSquaredError(mode=eval, name=nmse)
        >>> metric(torch.ones(2, 4), torch.ones(2, 4))
        >>> metric.value()
        {'eval/nmse': 0.0, 'eval/nmse_num_predictions': 8}
        >>> metric(torch.eye(2), torch.ones(2, 2))
        >>> metric.value()
        {'eval/nmse': 0.166666..., 'eval/nmse_num_predictions': 12}
        >>> metric.reset()
        >>> metric(torch.eye(2), torch.ones(2, 2))
        >>> metric.value()
        {'eval/nmse': 0.5, 'eval/nmse_num_predictions': 4}
    """

    def __init__(self, mode: str, name: str = "nmse") -> None:
        super().__init__(mode=mode, name=name)
        self._sum_squared_errors = 0.0
        self._sum_squared_targets = 0.0
        self._num_predictions = 0
        self.reset()

    def extra_repr(self) -> str:
        return f"mode={self._mode}, name={self._name}"

    def attach(self, engine: BaseEngine) -> None:
        super().attach(engine)
        engine.add_history(MinScalarHistory(name=self._metric_name))

    def forward(self, prediction: Tensor, target: Tensor) -> None:
        r"""Updates the mean squared error metric given a mini-batch of
        examples.

        Args:
        ----
            prediction (``torch.Tensor`` of shape
                ``(d0, d1, ..., dn)`` and type float or long):
                Specifies the tensor of predictions.
            target (``torch.Tensor`` of shape
                ``(d0, d1, ..., dn)`` and type float or long):
                Specifies the tensor of targets.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from gravitorch.models.metrics import NormalizedMeanSquaredError
            >>> metric = NormalizedMeanSquaredError("eval")
            >>> metric(torch.ones(2, 4), torch.ones(2, 4))
            >>> metric.value()
            {'eval/nmse': 0.0, 'eval/nmse_num_predictions': 8}
        """
        self._sum_squared_errors += mse_loss(
            prediction.float(), target.float(), reduction="sum"
        ).item()
        self._sum_squared_targets += target.pow(2).sum().item()
        self._num_predictions += target.numel()

    def reset(self) -> None:
        self._sum_squared_errors = 0.0
        self._sum_squared_targets = 0.0
        self._num_predictions = 0

    def value(self, engine: BaseEngine | None = None) -> dict:
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
