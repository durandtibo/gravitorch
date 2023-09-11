r"""This module defines a metric to compute the logarithm of the
hyperbolic cosine of the prediction error."""

from __future__ import annotations

__all__ = ["LogCoshError"]

import logging

from coola.utils import str_mapping
from torch import Tensor

from gravitorch.models.metrics.base_epoch import BaseStateEpochMetric
from gravitorch.models.metrics.state import BaseState, ErrorState
from gravitorch.nn.functional import log_cosh_loss

logger = logging.getLogger(__name__)


class LogCoshError(BaseStateEpochMetric):
    r"""Implements a metric to compute the logarithm of the hyperbolic
    cosine of the prediction error.

    Args:
    ----
        mode (str): Specifies the mode (e.g ``'train'`` or ``'eval'``).
        name (str, optional): Specifies the name of the metric. The
            name is used to log the metric results.
            Default: ``'log_cosh_err'``
        scale (float, optional): Specifies the scale factor.
            Default: ``1.0``
        state (``BaseState`` or dict, optional): Specifies the metric
            state or its configuration. If ``None``, ``ErrorState`` is
            instantiated. Default: ``None``

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from gravitorch.models.metrics import LogCoshError
        >>> metric = LogCoshError("eval")
        >>> metric
        LogCoshError(
          (mode): eval
          (name): log_cosh_err
          (scale): 1.0
          (state): ErrorState(num_predictions=0)
        )
        >>> metric(torch.ones(2, 4), torch.ones(2, 4))
        >>> metric.value()
        {'eval/log_cosh_err_mean': 0.0,
         'eval/log_cosh_err_min': 0.0,
         'eval/log_cosh_err_max': 0.0,
         'eval/log_cosh_err_sum': 0.0,
         'eval/log_cosh_err_num_predictions': 8}
        >>> metric(torch.eye(2), torch.ones(2, 2))
        >>> metric.value()
        {'eval/log_cosh_err_mean': 0.072296...,
         'eval/log_cosh_err_min': 0.0,
         'eval/log_cosh_err_max': 0.433780...,
         'eval/log_cosh_err_sum': 0.867561...,
         'eval/log_cosh_err_num_predictions': 12}
        >>> metric.reset()
        >>> metric(torch.eye(2), torch.ones(2, 2))
        >>> metric.value()
        {'eval/log_cosh_err_mean': 0.216890...,
         'eval/log_cosh_err_min': 0.0,
         'eval/log_cosh_err_max': 0.433780...,
         'eval/log_cosh_err_sum': 0.867561...,
         'eval/log_cosh_err_num_predictions': 4}
    """

    def __init__(
        self,
        mode: str,
        name: str = "log_cosh_err",
        scale: float = 1.0,
        state: BaseState | dict | None = None,
    ) -> None:
        super().__init__(mode=mode, name=name, state=state or ErrorState())
        if scale <= 0.0:
            raise ValueError(f"Incorrect scale {scale}. The scale has to be >0")
        self._scale = float(scale)

    def extra_repr(self) -> str:
        return str_mapping(
            {"mode": self._mode, "name": self._name, "scale": self._scale, "state": self._state}
        )

    def forward(self, prediction: Tensor, target: Tensor) -> None:
        r"""Updates the metric given a mini-batch of examples.

        Args:
        ----
            prediction (``torch.Tensor`` of shape
                ``(d0, d1, ..., dn)`` and type float or long):
                Specifies the predictions.
            target (``torch.Tensor`` of shape
                ``(d0, d1, ..., dn)`` and type float or long):
                Specifies the target tensor.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from gravitorch.models.metrics import LogCoshError
            >>> metric = LogCoshError("eval")
            >>> metric(torch.ones(2, 4), torch.ones(2, 4))
            >>> metric.value()
            {'eval/log_cosh_err_mean': 0.0,
             'eval/log_cosh_err_min': 0.0,
             'eval/log_cosh_err_max': 0.0,
             'eval/log_cosh_err_sum': 0.0,
             'eval/log_cosh_err_num_predictions': 8}
        """
        self._state.update(log_cosh_loss(prediction, target, reduction="none", scale=self._scale))
