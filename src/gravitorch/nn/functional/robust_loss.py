__all__ = [
    "asinh_mse_loss",
    "log_cosh_loss",
    "msle_loss",
    "relative_mse_loss",
    "relative_smooth_l1_loss",
    "symlog_mse_loss",
    "symmetric_relative_smooth_l1_loss",
]

from torch import Tensor
from torch.nn import functional as F

from gravitorch.nn.functional.loss_helpers import basic_loss_reduction
from gravitorch.utils.tensor.math_ops import symlog


def msle_loss(prediction: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    r"""Computes the mean squared logarithmic error (MSLE).

    This loss is best to use when targets having exponential growth,
    such as population counts, average sales of a commodity over a
    span of years etc. Note that this loss penalizes an
    under-predicted estimate greater than an over-predicted estimate.

    Note: this loss only works with positive values (0 included).

    Args:
        prediction (``torch.Tensor`` of type float and shape
            ``(d0, d1, ..., dn)``): Specifies the predictions.
        target (``torch.Tensor`` of type float and shape
            ``(d0, d1, ..., dn)``): Specifies the target values.
        reduction (string, optional): Specifies the reduction to apply
            to the output: ``'none'`` | ``'mean'`` | ``'sum'``.
            ``'none'``: no reduction will be applied, ``'mean'``:
            the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be
            summed. Default: ``'mean'``

    Returns:
        ``torch.Tensor`` of type float: The mean squared logarithmic
            error. The shape of the tensor depends on the reduction
            strategy.
    """
    return F.mse_loss(prediction.log1p(), target.log1p(), reduction=reduction)


def asinh_mse_loss(prediction: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    r"""Computes the mean squared error (MSE) on the arcsinh transformed
    prediction and target.

    It is a generalization of mean squared logarithmic error (MSLE)
    that works for real values. The ``arcsinh`` transformation is used
    instead of ``log1p`` because ``arcsinh`` works on negative values.

    Args:
        prediction (``torch.Tensor`` of type float and shape
            ``(d0, d1, ..., dn)``): Specifies the predictions.
        target (``torch.Tensor`` of type float and shape
            ``(d0, d1, ..., dn)``): Specifies the target values.
        reduction (string, optional): Specifies the reduction to
            apply to the output: ``'none'`` | ``'mean'`` | ``'sum'``.
            ``'none'``: no reduction will be applied, ``'mean'``: the
            sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be
            summed. Default: ``'mean'``

    Returns:
        ``torch.Tensor`` of type float: The mean squared error (MSE)
            on the arcsinh transformed prediction and target. The
            shape of the tensor depends on the reduction strategy.
    """
    return F.mse_loss(prediction.asinh(), target.asinh(), reduction=reduction)


def symlog_mse_loss(prediction: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    r"""Computes the mean squared error (MSE) on the symlog transformed
    prediction and target.

    It is a generalization of mean squared logarithmic error (MSLE)
    that works for real values. The ``symlog`` transformation is used
    instead of ``log1p`` because ``symlog`` works on negative values.

    Args:
        prediction (``torch.Tensor`` of type float and shape
            ``(d0, d1, ..., dn)``): Specifies the predictions.
        target (``torch.Tensor`` of type float and shape
            ``(d0, d1, ..., dn)``): Specifies the target values.
        reduction (string, optional): Specifies the reduction to
            apply to the output: ``'none'`` | ``'mean'`` | ``'sum'``.
            ``'none'``: no reduction will be applied, ``'mean'``: the
            sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be
            summed. Default: ``'mean'``

    Returns:
        ``torch.Tensor`` of type float: The mean squared error (MSE)
            on the symlog transformed prediction and target. The shape
            of the tensor depends on the reduction strategy.
    """
    return F.mse_loss(symlog(prediction), symlog(target), reduction=reduction)


def relative_mse_loss(
    prediction: Tensor, target: Tensor, reduction: str = "mean", eps: float = 1e-8
):
    r"""Computes the relative mean squared error loss.

    Args:
        prediction (``torch.Tensor`` of type float and shape
            ``(d0, d1, ..., dn)``): Specifies the predictions.
        target (``torch.Tensor`` of type float and shape
            ``(d0, d1, ..., dn)``): Specifies the target values.
        reduction (string, optional): Specifies the reduction to
            apply to the output: ``'none'`` | ``'mean'`` | ``'sum'``.
            ``'none'``: no reduction will be applied, ``'mean'``: the
            sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be
            summed. Default: ``'mean'``
        eps (float, optional): Specifies an arbitrary small strictly
            positive number to avoid undefined results when the target
            is zero. Default: ``1e-8``

    Returns:
        ``torch.Tensor`` of type float: The relative mean squared error.
            The shape depends on the reduction strategy.

    Example usage:

    .. code-block:: python

        >>> import torch
        >>> from gravitorch.nn.functional import relative_mse_loss
        >>> loss = relative_mse_loss(torch.randn(3, 5, requires_grad=True), torch.randn(3, 5))
        >>> loss.backward()
    """
    return basic_loss_reduction(
        F.mse_loss(prediction, target, reduction="none").div(target.detach().pow(2).clamp(min=eps)),
        reduction=reduction,
    )


def relative_smooth_l1_loss(
    prediction: Tensor,
    target: Tensor,
    reduction: str = "mean",
    beta: float = 1.0,
    eps: float = 1e-8,
):
    r"""Computes the relative smooth l1 loss.

    Args:
        prediction (``torch.Tensor`` of type float and shape
            ``(d0, d1, ..., dn)``): Specifies the predictions.
        target (``torch.Tensor`` of type float and shape
            ``(d0, d1, ..., dn)``): Specifies the target values.
        reduction (string, optional): Specifies the reduction to
            apply to the output: ``'none'`` | ``'mean'`` | ``'sum'``.
            ``'none'``: no reduction will be applied, ``'mean'``: the
            sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be
            summed. Default: ``'mean'``
        beta (float, optional): Specifies the threshold at which to
            change between L1 and L2 loss. The value must be
            non-negative. Default: ``1.0``
        eps (float, optional): Specifies an arbitrary small strictly
            positive number to avoid undefined results when the target
            is zero. Default: ``1e-8``

    Returns:
        ``torch.Tensor`` of type float: The relative smooth l1 loss.
            The shape depends on the reduction strategy.

    Example usage:

    .. code-block:: python

        >>> import torch
        >>> from gravitorch.nn.functional import relative_smooth_l1_loss
        >>> loss = relative_smooth_l1_loss(torch.randn(3, 5, requires_grad=True), torch.randn(3, 5))
        >>> loss.backward()
    """
    return basic_loss_reduction(
        F.smooth_l1_loss(prediction, target, reduction="none", beta=beta).div(
            target.detach().abs().clamp(min=eps)
        ),
        reduction=reduction,
    )


def symmetric_relative_smooth_l1_loss(
    prediction: Tensor,
    target: Tensor,
    reduction: str = "mean",
    beta: float = 1.0,
    eps: float = 1e-8,
):
    r"""Computes the symmetric relative smooth l1 loss.

    Args:
        prediction (``torch.Tensor`` of type float and shape
            ``(d0, d1, ..., dn)``): Specifies the predictions.
        target (``torch.Tensor`` of type float and shape
            ``(d0, d1, ..., dn)``): Specifies the target values.
        reduction (string, optional): Specifies the reduction to
            apply to the output: ``'none'`` | ``'mean'`` | ``'sum'``.
            ``'none'``: no reduction will be applied, ``'mean'``: the
            sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be
            summed. Default: ``'mean'``
        beta (float, optional): Specifies the threshold at which to
            change between L1 and L2 loss. The value must be
            non-negative. Default: ``1.0``
        eps (float, optional): Specifies an arbitrary small strictly
            positive number to avoid undefined results when the target
            is zero. Default: ``1e-8``

    Returns:
        ``torch.Tensor`` of type float: The symmetric relative smooth
            l1 loss. The shape depends on the reduction strategy.

    Example usage:

    .. code-block:: python

        >>> import torch
        >>> from gravitorch.nn.functional import symmetric_relative_smooth_l1_loss
        >>> loss = symmetric_relative_smooth_l1_loss(
        ...     torch.randn(3, 5, requires_grad=True), torch.randn(3, 5)
        ... )
        >>> loss.backward()
    """
    return basic_loss_reduction(
        F.smooth_l1_loss(prediction, target, reduction="none", beta=beta).div(
            target.abs().add(prediction.abs()).detach().mul(0.5).clamp(min=eps)
        ),
        reduction=reduction,
    )


def log_cosh_loss(
    prediction: Tensor,
    target: Tensor,
    reduction: str = "mean",
    scale: float = 1.0,
):
    r"""Computes the logarithm of the hyperbolic cosine of the prediction error.

    Args:
        prediction (``torch.Tensor`` of type float and shape
            ``(d0, d1, ..., dn)``): Specifies the predictions.
        target (``torch.Tensor`` of type float and shape
            ``(d0, d1, ..., dn)``): Specifies the target values.
        reduction (string, optional): Specifies the reduction to
            apply to the output: ``'none'`` | ``'mean'`` | ``'sum'``.
            ``'none'``: no reduction will be applied, ``'mean'``: the
            sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be
            summed. Default: ``'mean'``
        scale (float, optional): Specifies the scale factor.
            Default: ``1.0``

    Returns:
        ``torch.Tensor`` of type float: the logarithm of the
            hyperbolic cosine of the prediction error.

    Example usage:

    .. code-block:: python

        >>> import torch
        >>> from gravitorch.nn.functional import log_cosh_loss
        >>> loss = log_cosh_loss(torch.randn(3, 5, requires_grad=True), torch.randn(3, 5))
        >>> loss.backward()
    """
    return basic_loss_reduction(
        target.sub(prediction).div(scale).cosh().log(),
        reduction=reduction,
    )
