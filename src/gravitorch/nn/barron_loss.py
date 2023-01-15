__all__ = ["AsinhBarronRobustLoss", "BarronRobustLoss"]

from typing import Optional

from torch import Tensor
from torch.nn import Module

from gravitorch.nn.functional import asinh_barron_robust_loss, barron_robust_loss
from gravitorch.nn.functional.loss_helpers import VALID_REDUCTIONS


class BarronRobustLoss(Module):
    r"""Implements the Barron robust loss function.

    Based on the paper:

        A General and Adaptive Robust Loss Function
        Jonathan T. Barron
        CVPR 2019 (https://arxiv.org/abs/1701.03077)

    Args:
        alpha (float, optional): Specifies the shape parameter that
            controls the robustness of the loss. Default: ``2.0``
        scale (float, optional): Specifies the scale parameter that
            controls the size of the loss’s quadratic bowl near 0.
            Default: ``1.0``
        max_value (float or ``None``, optional): Specifies the max
            value to clip the loss before to compute the reduction.
            ``None`` means no clipping is used. Default: ``None``
        reduction (string, optional): Specifies the reduction to
            apply to the output: ``'none'`` | ``'mean'`` | ``'sum'``.
            ``'none'``: no reduction will be applied, ``'mean'``: the
            sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be
            summed. Default: ``"mean"``
    """

    def __init__(
        self,
        alpha: float = 2.0,
        scale: float = 1.0,
        max_value: Optional[float] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self._alpha = float(alpha)
        if scale <= 0:
            raise ValueError(f"scale has to be greater than 0 but received {scale}")
        self._scale = float(scale)
        self._max_value = max_value

        if reduction not in VALID_REDUCTIONS:
            raise ValueError(
                f"Incorrect reduction: {reduction}. The valid reductions are: {VALID_REDUCTIONS}"
            )
        self.reduction = reduction

    def extra_repr(self) -> str:
        return (
            f"alpha={self._alpha}, scale={self._scale}, max_value={self._max_value}, "
            f"reduction={self.reduction}"
        )

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        r"""Computes the loss values and reduces them.

        Args:
            prediction (``torch.Tensor``): Specifies the predictions.
            target (``torch.Tensor``): Specifies the targets.

        Returns:
            ``torch.Tensor``: The computed loss value.
        """
        return barron_robust_loss(
            prediction=prediction,
            target=target,
            alpha=self._alpha,
            scale=self._scale,
            max_value=self._max_value,
            reduction=self.reduction,
        )


class AsinhBarronRobustLoss(BarronRobustLoss):
    r"""Implements the Barron robust loss function where the inverse hyperbolic
    sine (arcsinh) transformation is applied on the prediction and target."""

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        r"""Computes the loss values and reduces them.

        Args:
            prediction (``torch.Tensor``): Specifies the predictions.
            target (``torch.Tensor``): Specifies the targets.

        Returns:
            ``torch.Tensor``: The computed loss value.
        """
        return asinh_barron_robust_loss(
            prediction=prediction,
            target=target,
            alpha=self._alpha,
            scale=self._scale,
            max_value=self._max_value,
            reduction=self.reduction,
        )
