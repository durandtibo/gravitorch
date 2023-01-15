r"""This module defines some absolute relative error metrics."""

__all__ = ["AbsoluteRelativeError", "SymmetricAbsoluteRelativeError"]

import logging
from typing import Union

from torch import Tensor

from gravitorch.models.metrics.base_epoch import BaseStateEpochMetric
from gravitorch.models.metrics.state import BaseState, ErrorState
from gravitorch.nn.functional import (
    absolute_relative_error,
    symmetric_absolute_relative_error,
)

logger = logging.getLogger(__name__)


class AbsoluteRelativeError(BaseStateEpochMetric):
    r"""Implements the absolute relative error metric.

    Args:
        mode (str): Specifies the mode (e.g ``'train'`` or ``'eval'``).
        name (str, optional): Specifies the name of the metric. The
            name is used to log the metric results.
            Default: ``'abs_rel_err'``
        eps (float, optional): Specifies an arbitrary small strictly
            positive number to avoid undefined results when the target
            is zero. Default: ``1e-8``

    Example usage:

    .. code-block:: python

        >>> import torch
        >>> from gravitorch.models.metrics import AbsoluteRelativeError
        >>> metric = AbsoluteRelativeError('eval')
        >>> metric
        AbsoluteRelativeErrorMetric(mode=eval, name=abs_rel_err, eps=1e-08)
        >>> metric(torch.ones(2, 4), torch.ones(2, 4))
        >>> metric.value()
        {'eval/abs_rel_err_mean': 0.0,
         'eval/abs_rel_err_min': 0.0,
         'eval/abs_rel_err_max': 0.0,
         'eval/abs_rel_err_sum': 0.0,
         'eval/abs_rel_err_num_predictions': 8}
        >>> metric(torch.eye(2), torch.ones(2, 2))
        >>> metric.value()
        {'eval/abs_rel_err_mean': 0.16666666666666666,
         'eval/abs_rel_err_min': 0.0,
         'eval/abs_rel_err_max': 1.0,
         'eval/abs_rel_err_sum': 2.0,
         'eval/abs_rel_err_num_predictions': 12}
        >>> metric.reset()
        >>> metric(torch.eye(2), torch.ones(2, 2))
        >>> metric.value()
        {'eval/abs_rel_err_mean': 0.5,
         'eval/abs_rel_err_min': 0.0,
         'eval/abs_rel_err_max': 1.0,
         'eval/abs_rel_err_sum': 2.0,
         'eval/abs_rel_err_num_predictions': 4}
    """

    def __init__(
        self,
        mode: str,
        name: str = "abs_rel_err",
        eps: float = 1e-8,
        state: Union[BaseState, dict, None] = None,
    ):
        super().__init__(mode=mode, name=name, state=state or ErrorState())
        if eps <= 0:
            raise ValueError(
                f"Incorrect eps ({eps}). eps has to be an arbitrary small strictly "
                f"positive number to avoid undefined results when the target is zero."
            )
        self._eps = float(eps)

    def forward(self, prediction: Tensor, target: Tensor) -> None:
        r"""Updates the mean absolute percentage error metric given a mini-batch
        of examples.

        Args:
            prediction (``torch.Tensor`` of shape
                ``(d0, d1, ..., dn)`` and type float or long):
                Specifies the predictions.
            target (``torch.Tensor`` with same shape and data type as
                ``prediction``): Specifies the target tensor.
        """
        self._state.update(absolute_relative_error(prediction, target, eps=self._eps))


class SymmetricAbsoluteRelativeError(BaseStateEpochMetric):
    r"""Implements the symmetric absolute relative error (SARE) metric.

    This metric tracks the mean, maximum and minimum absolute
    relative error values.

    Args:
        mode (str): Specifies the mode (e.g ``'train'`` or ``'eval'``).
        name (str, optional): Specifies the name of the metric. The
            name is used to log the metric results.
            Default: ``'sym_abs_rel_err'``
        eps (float, optional): Specifies an arbitrary small strictly
            positive number to avoid undefined results when the target
            is zero. Default: ``1e-8``

    Example usage:

    .. code-block:: python

        >>> import torch
        >>> from gravitorch.models.metrics import SymmetricAbsoluteRelativeError
        >>> metric = SymmetricAbsoluteRelativeError('eval')
        >>> metric
        SymmetricAbsoluteRelativeErrorMetric(mode=eval, name=sym_abs_rel_err, eps=1e-08)
        >>> metric(torch.ones(2, 4), torch.ones(2, 4))
        >>> metric.value()
        {'eval/sym_abs_rel_err_mean': 0.0,
         'eval/sym_abs_rel_err_min': 0.0,
         'eval/sym_abs_rel_err_max': 0.0,
         'eval/sym_abs_rel_err_sum': 0.0,
         'eval/sym_abs_rel_err_num_predictions': 8}
        >>> metric(torch.eye(2), torch.ones(2, 2))
        >>> metric.value()
        {'eval/sym_abs_rel_err_mean': 0.3333333333333333,
         'eval/sym_abs_rel_err_min': 0.0,
         'eval/sym_abs_rel_err_max': 2.0,
         'eval/sym_abs_rel_err_sum': 4.0,
         'eval/sym_abs_rel_err_num_predictions': 12}
        >>> metric.reset()
        >>> metric(torch.eye(2), torch.ones(2, 2))
        >>> metric.value()
        {'eval/sym_abs_rel_err_mean': 1.0,
         'eval/sym_abs_rel_err_min': 0.0,
         'eval/sym_abs_rel_err_max': 2.0,
         'eval/sym_abs_rel_err_sum': 4.0,
         'eval/sym_abs_rel_err_num_predictions': 4}
    """

    def __init__(
        self,
        mode: str,
        name: str = "sym_abs_rel_err",
        eps: float = 1e-8,
        state: Union[BaseState, dict, None] = None,
    ):
        super().__init__(mode=mode, name=name, state=state or ErrorState())
        if eps <= 0:
            raise ValueError(
                f"Incorrect eps ({eps}). eps has to be an arbitrary small strictly "
                f"positive number to avoid undefined results when the target is zero."
            )
        self._eps = float(eps)

    def forward(self, prediction: Tensor, target: Tensor) -> None:
        r"""Updates the mean absolute percentage error metric given a mini-batch
        of examples.

        Args:
            prediction (``torch.Tensor`` of shape
                ``(d0, d1, ..., dn)`` and type float or long):
                Specifies the predictions.
            target (``torch.Tensor`` of shape
                ``(d0, d1, ..., dn)`` and type float or long):
                Specifies the target tensor.
        """
        self._state.update(symmetric_absolute_relative_error(prediction, target, eps=self._eps))
