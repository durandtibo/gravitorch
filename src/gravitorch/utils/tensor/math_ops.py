r"""This module implements some utility functions to use ``torch.Tensor``s."""

__all__ = ["isymlog", "isymlog_", "safeexp", "safelog", "symlog", "symlog_", "scalable_quantile"]

import numpy as np
import torch
from torch import Tensor


def scalable_quantile(tensor: Tensor, q: Tensor, method: str = "linear") -> Tensor:
    r"""Implements a scalable function to compute the ``q``-th quantiles.

    Note: this function uses numpy to compute the ``q``-th quantiles
    because PyTorch has a limit to 16M items.
    https://github.com/pytorch/pytorch/issues/64947

    Args:
        tensor (``torch.Tensor``): Specifies the tensor of values.
        q (``torch.Tensor`` of type float and shape
            ``(num_q_values,)``): Specifies the ``q``-values in
            the range ``[0, 1]``.
        method (str, optional): Specifies the interpolation
            method to use when the desired quantile lies between
            two data points. Can be ``'linear'``, ``'lower'``,
            ``'higher'``, ``'midpoint'`` and ``'nearest'``.
            Default: ``'linear'``.

    Returns:
        ``torch.Tensor`` of shape  ``(num_q_values,)`` and type float:
            The ``q``-th quantiles.
    """
    return torch.from_numpy(
        np.quantile(tensor.detach().cpu().numpy(), q=q.numpy(), method=method)
    ).float()


def safeexp(tensor: Tensor, max_value: float = 20.0) -> Tensor:
    r"""Computes safely the exponential of the elements.

    The values that are higher than the specified minimum value are
    set to this maximum value. Using a not too large positive value
    leads to an output tensor without Inf.

    Args:
        tensor (``torch.Tensor``): Specifies the input tensor.
        max_value (float, optional): Specifies the maximum value.
            Default: ``20.0``

    Returns:
        ``torch.Tensor``: A new tensor with the exponential of the
            elements.
    """
    return tensor.clamp(max=max_value).exp()


def safelog(tensor: Tensor, min_value: float = 1e-8) -> Tensor:
    r"""Computes safely the logarithm natural logarithm of the elements.

    The values that are lower than the specified minimum value are set
    to this minimum value. Using a small positive value leads to an
    output tensor without NaN or Inf.

    Args:
        tensor (``torch.Tensor``): Specifies the input tensor.
        min_value (float, optional): Specifies the minimum value.
            Default: ``1e-8``

    Returns:
        ``torch.Tensor``: A new tensor with the natural logarithm
            of the elements.
    """
    return tensor.clamp(min=min_value).log()


def symlog(tensor: Tensor) -> Tensor:
    r"""Computes the symmetric logarithm natural logarithm of the elements.

    Note this transformation supports negative values.

    Args:
        tensor (``torch.Tensor``): Specifies the input tensor.

    Returns:
        ``torch.Tensor``: A new tensor with the symmetric natural
            logarithm of the elements.
    """
    return tensor.sign().mul(tensor.abs().log1p())


def symlog_(tensor: Tensor) -> None:
    r"""Computes the symmetric logarithm natural of the elements.

    In-place version of ``symlog``.

    Note this transformation supports negative values.

    Args:
        tensor (``torch.Tensor``): Specifies the input tensor.
    """
    sign = tensor.sign()
    tensor.abs_()
    tensor.log1p_()
    tensor.mul_(sign)


def isymlog(tensor: Tensor, max_value: float = 10) -> Tensor:
    r"""Computes the inverse of the symmetric logarithm natural of the elements.

    Args:
        tensor (``torch.Tensor``): Specifies the input tensor.

    Returns:
        ``torch.Tensor``: A new tensor with the inverse of the
            symmetric natural logarithm of the elements.
    """
    sign = tensor.sign()
    return sign.mul(tensor.mul(sign).clamp(max=max_value).exp().sub(1))


def isymlog_(tensor: Tensor, max_value: float = 10) -> None:
    r"""Computes the inverse of the symmetric logarithm natural of the elements.

    In-place version of ``isymlog``.

    Args:
        tensor (``torch.Tensor``): Specifies the input tensor.
    """
    sign = tensor.sign()
    tensor.mul_(sign)
    tensor.clamp_(max=max_value)
    tensor.exp_()
    tensor.sub_(1)
    tensor.mul_(sign)
