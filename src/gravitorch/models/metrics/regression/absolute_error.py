r"""This module defines the absolute error metric."""

__all__ = ["AbsoluteError"]

import logging
from typing import Union

from torch import Tensor

from gravitorch.models.metrics.base_epoch import BaseStateEpochMetric
from gravitorch.models.metrics.state import BaseState, ErrorState
from gravitorch.nn.functional import absolute_error

logger = logging.getLogger(__name__)


class AbsoluteError(BaseStateEpochMetric):
    r"""Implements the absolute error metric.

    Args:
        mode (str): Specifies the mode (e.g ``'train'`` or ``'eval'``).
        name (str, optional): Specifies the name of the metric. The
            name is used to log the metric results.
            Default: ``'abs_err'``
        state (``BaseState`` or dict, optional): Specifies the metric
            state or its configuration. If ``None``, ``ErrorState`` is
            instantiated. Default: ``None``

    Example usage:

    .. code-block:: python

        >>> import torch
        >>> from gravitorch.models.metrics import AbsoluteError
        >>> metric = AbsoluteError('eval')
        >>> metric
        AbsoluteErrorMetric(mode=eval, name=abs_err)
        >>> metric(torch.ones(2, 4), torch.ones(2, 4))
        >>> metric.value()
        {'eval/abs_err_mean': 0.0,
         'eval/abs_err_min': 0.0,
         'eval/abs_err_max': 0.0,
         'eval/abs_err_sum': 0.0,
         'eval/abs_err_num_predictions': 8}
        >>> metric(torch.eye(2), torch.ones(2, 2))
        >>> metric.value()
        {'eval/abs_err_mean': 0.16666666666666666,
         'eval/abs_err_min': 0.0,
         'eval/abs_err_max': 1.0,
         'eval/abs_err_sum': 2.0,
         'eval/abs_err_num_predictions': 12}
        >>> metric.reset()
        >>> metric(torch.eye(2), torch.ones(2, 2))
        >>> metric.value()
        {'eval/abs_err_mean': 0.5,
         'eval/abs_err_min': 0.0,
         'eval/abs_err_max': 1.0,
         'eval/abs_err_sum': 2.0,
         'eval/abs_err_num_predictions': 4}
    """

    def __init__(
        self,
        mode: str,
        name: str = "abs_err",
        state: Union[BaseState, dict, None] = None,
    ):
        super().__init__(mode=mode, name=name, state=state or ErrorState())

    def forward(self, prediction: Tensor, target: Tensor) -> None:
        r"""Updates the mean absolute error metric given a mini-batch of
        examples.

        Args:
            prediction (``torch.Tensor`` of shape
                ``(d0, d1, ..., dn)`` and type float or long):
                Specifies the predictions.
            target (``torch.Tensor`` of shape
                ``(d0, d1, ..., dn)`` and type float or long):
                Specifies the target tensor.
        """
        self._state.update(absolute_error(prediction, target))
