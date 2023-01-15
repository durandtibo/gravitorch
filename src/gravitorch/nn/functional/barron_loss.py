__all__ = ["asinh_barron_robust_loss", "barron_robust_loss"]

import math
from typing import Optional

from torch import Tensor

from gravitorch.nn.functional.loss_helpers import basic_loss_reduction


def barron_robust_loss(
    prediction: Tensor,
    target: Tensor,
    alpha: float = 2.0,
    scale: float = 1.0,
    max_value: Optional[float] = None,
    reduction: str = "mean",
) -> Tensor:
    r"""Computes the Barron robust loss.

    Based on the paper:

        A General and Adaptive Robust Loss Function
        Jonathan T. Barron
        CVPR 2019 (https://arxiv.org/abs/1701.03077)

    Args:
        prediction (``torch.Tensor`` of type float and shape
            ``(d0, d1, ..., dn)``): Specifies the predictions.
        target (``torch.Tensor`` of type float and shape
            ``(d0, d1, ..., dn)``): Specifies the target values.
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

    Returns:
        ``torch.Tensor`` of type float: The computed loss. The shape
            of the tensor depends on the reduction strategy.
    """
    squared_error = prediction.sub(target).div(scale).pow(2)
    if alpha == 2:
        loss = squared_error
    elif alpha == 0:
        loss = squared_error.mul(0.5).add(1).log()
    else:
        alpha2 = math.fabs(alpha - 2)
        loss = squared_error.div(alpha2).add(1).pow(alpha / 2).sub(1).mul(alpha2 / alpha)
    if max_value is not None:
        loss = loss.clamp(max=max_value)
    return basic_loss_reduction(loss, reduction)


def asinh_barron_robust_loss(
    prediction: Tensor,
    target: Tensor,
    alpha: float = 2.0,
    scale: float = 1.0,
    max_value: Optional[float] = None,
    reduction: str = "mean",
) -> Tensor:
    r"""Computes the Barron loss on the inverse hyperbolic sine (arcsinh)
    transformed prediction and target.

    Args:
        prediction (``torch.Tensor`` of type float and shape
            ``(d0, d1, ..., dn)``): Specifies the predictions.
        target (``torch.Tensor`` of type float and shape
            ``(d0, d1, ..., dn)``): Specifies the target values.
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

    Returns:
        ``torch.Tensor`` of type float: The computed loss. The shape
            of the tensor depends on the reduction strategy.
    """
    return barron_robust_loss(
        prediction=prediction.asinh(),
        target=target.asinh(),
        alpha=alpha,
        scale=scale,
        max_value=max_value,
        reduction=reduction,
    )
