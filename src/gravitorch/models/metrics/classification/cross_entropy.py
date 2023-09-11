from __future__ import annotations

__all__ = ["CategoricalCrossEntropy"]

import logging

from torch import Tensor
from torch.nn.functional import cross_entropy

from gravitorch.models.metrics.base_epoch import BaseStateEpochMetric
from gravitorch.models.metrics.state import BaseState, MeanErrorState

logger = logging.getLogger(__name__)


class CategoricalCrossEntropy(BaseStateEpochMetric):
    r"""Implements a metric to compute the categorical cross-entropy.

    Args:
    ----
        mode (str): Specifies the mode.
        name (str, optional): Specifies the name used to log the
            metric. Default: ``'cat_acc'``
        state (``BaseState`` or dict or ``None``, optional): Specifies
            the metric state or its configuration. If ``None``,
            ``MeanErrorState`` is instantiated. Default: ``None``

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from gravitorch.models.metrics import CategoricalCrossEntropy
        >>> metric = CategoricalCrossEntropy("eval")
        >>> metric
        CategoricalCrossEntropy(
          (mode): eval
          (name): cat_ce
          (state): MeanErrorState(num_predictions=0)
        )
        >>> metric(torch.eye(4), torch.arange(4))
        >>> metric.value()  # doctest:+ELLIPSIS
        {'eval/cat_ce_mean': 0.743668..., 'eval/cat_ce_num_predictions': 4}
        >>> metric(torch.ones(2, 3), torch.ones(2))
        >>> metric.value()  # doctest:+ELLIPSIS
        {'eval/cat_ce_mean': 0.861983..., 'eval/cat_ce_num_predictions': 6}
        >>> metric.reset()
        >>> metric(torch.ones(2, 3), torch.ones(2))
        >>> metric.value()  # doctest:+ELLIPSIS
        {'eval/cat_ce_mean': 1.098612..., 'eval/cat_ce_num_predictions': 2}
    """

    def __init__(
        self,
        mode: str,
        name: str = "cat_ce",
        state: BaseState | dict | None = None,
    ) -> None:
        super().__init__(mode=mode, name=name, state=state or MeanErrorState())

    def forward(self, prediction: Tensor, target: Tensor) -> None:
        r"""Updates the metric given a mini-batch of examples.

        Args:
        ----
            prediction (``torch.Tensor`` of shape
                ``(d0, d1, ..., dn, num_classes)`` and type float):
                Specifies the predictions.
            target (``torch.Tensor`` of shape
                ``(d0, d1, ..., dn)`` or ``(d0, d1, ..., dn, 1)`` and
                type long or float): Specifies the categorical
                targets. The values have to be in
                ``{0, 1, ..., num_classes-1}``.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from gravitorch.models.metrics import CategoricalCrossEntropy
            >>> metric = CategoricalCrossEntropy("eval")
            >>> metric(torch.eye(4), torch.arange(4))
            >>> metric.value()  # doctest:+ELLIPSIS
            {'eval/cat_ce_mean': 0.743668..., 'eval/cat_ce_num_predictions': 4}
        """
        self._state.update(
            cross_entropy(
                prediction.flatten(start_dim=0, end_dim=-2),
                target.flatten().long(),
                reduction="none",
            )
        )
