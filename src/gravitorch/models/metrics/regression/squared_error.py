r"""This module defines the squared logarithmic error (SLE) metric."""

__all__ = ["RootMeanSquaredError", "SquaredError"]

import logging
from typing import Union

from torch import Tensor
from torch.nn.functional import mse_loss

from gravitorch.models.metrics.base_epoch import BaseStateEpochMetric
from gravitorch.models.metrics.state import BaseState, ErrorState, RootMeanErrorState

logger = logging.getLogger(__name__)


class SquaredError(BaseStateEpochMetric):
    r"""Implements the squared error metric.

    Args:
        mode (str): Specifies the mode (e.g. train or eval).
        name (str, optional): Specifies the name of the metric. The
            name is used to log the metric results.
            Default: ``'sq_err'``
        state (``BaseState`` or dict, optional): Specifies the metric
            state or its configuration. If ``None``, ``ErrorState`` is
            instantiated. Default: ``None``

    Example usage:

    .. code-block:: python

        >>> import torch
        >>> from gravitorch.models.metrics import SquaredError
        >>> metric = SquaredError('eval')
        >>> metric(torch.ones(2, 4), torch.ones(2, 4))
        >>> metric.value()
        {'eval/sq_err_mean': 0.0,
         'eval/sq_err_max': 0.0,
         'eval/sq_err_min': 0.0,
         'eval/sq_err_sum': 0.0,
         'eval/sq_err_num_predictions': 8}
        >>> metric(torch.eye(2), torch.ones(2, 2))
        >>> metric.value()
        {'eval/sq_err_mean': 0.16666666666666666,
         'eval/sq_err_max': 1.0,
         'eval/sq_err_min': 0.0,
         'eval/sq_err_sum': 2.0,
         'eval/sq_err_num_predictions': 12}
        >>> metric.reset()
        >>> metric(torch.eye(2), torch.ones(2, 2))
        >>> metric.value()
        {'eval/sq_err_mean': 0.5,
         'eval/sq_err_max': 1.0,
         'eval/sq_err_min': 0.0,
         'eval/sq_err_sum': 2.0,
         'eval/sq_err_num_predictions': 4}
    """

    def __init__(
        self,
        mode: str,
        name: str = "sq_err",
        state: Union[BaseState, dict, None] = None,
    ):
        super().__init__(mode=mode, name=name, state=state or ErrorState())

    def forward(self, prediction: Tensor, target: Tensor) -> None:
        r"""Updates the squared logarithmic error metric given a mini-batch of
        examples.

        Args:
            prediction (``torch.Tensor`` of shape
                ``(d0, d1, ..., dn)`` and type float or long):
                Specifies the predictions.
            target (``torch.Tensor`` of shape
                ``(d0, d1, ..., dn)`` and type float or long):
                Specifies the target tensor.
        """
        self._state.update(mse_loss(prediction.float(), target.float(), reduction="none"))


class RootMeanSquaredError(SquaredError):
    r"""Implements the squared error metric.

    Args:
        mode (str): Specifies the mode (e.g. train or eval).
        name (str, optional): Specifies the name of the metric. The
            name is used to log the metric results.
            Default: ``'sq_err'``
        track_num_predictions (bool, optional): If ``True``, the state
            tracks and returns the number of predictions.
            Default: ``True``

    Example usage:

    .. code-block:: python

        >>> import torch
        >>> from gravitorch.models.metrics import RootMeanSquaredError
        >>> metric = RootMeanSquaredError('eval')
        >>> metric(torch.ones(2, 4), torch.ones(2, 4))
        >>> metric.value()
        {'eval/rmse_root_mean': 0.0, 'eval/rmse_num_predictions': 8}
        >>> metric(torch.eye(2), torch.ones(2, 2))
        >>> metric.value()
        {'eval/rmse_root_mean': 0.408248290463863, 'eval/rmse_num_predictions': 12}
        >>> metric.reset()
        >>> metric(torch.eye(2), torch.ones(2, 2))
        >>> metric.value()
        {'eval/rmse_root_mean': 0.7071067811865476, 'eval/rmse_num_predictions': 4}
    """

    def __init__(
        self,
        mode: str,
        name: str = "rmse",
        track_num_predictions: bool = True,
    ):
        super().__init__(
            mode=mode,
            name=name,
            state=RootMeanErrorState(track_num_predictions=track_num_predictions),
        )
