r"""This module implements some utility functions to use
``torch.Tensor``s."""

from __future__ import annotations

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
    ----
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
    -------
        ``torch.Tensor`` of shape  ``(num_q_values,)`` and type float:
            The ``q``-th quantiles.

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from gravitorch.utils.tensor import scalable_quantile
        >>> scalable_quantile(torch.arange(1001), q=torch.tensor([0.1, 0.9]))
        tensor([100., 900.])
    """
    return torch.from_numpy(
        np.quantile(tensor.detach().cpu().numpy(), q=q.numpy(), method=method)
    ).to(dtype=torch.float, device=tensor.device)


def safeexp(tensor: Tensor, max_value: float = 20.0) -> Tensor:
    r"""Computes safely the exponential of the elements.

    The values that are higher than the specified minimum value are
    set to this maximum value. Using a not too large positive value
    leads to an output tensor without Inf.

    Args:
    ----
        tensor (``torch.Tensor``): Specifies the input tensor.
        max_value (float, optional): Specifies the maximum value.
            Default: ``20.0``

    Returns:
    -------
        ``torch.Tensor``: A new tensor with the exponential of the
            elements.

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from gravitorch.utils.tensor import safeexp
        >>> safeexp(torch.tensor([1.0, 10.0, 100.0, 1000.0]))
        tensor([2.7183e+00, 2.2026e+04, 4.8517e+08, 4.8517e+08])
    """
    return tensor.clamp(max=max_value).exp()


def safelog(tensor: Tensor, min_value: float = 1e-8) -> Tensor:
    r"""Computes safely the logarithm natural logarithm of the elements.

    The values that are lower than the specified minimum value are set
    to this minimum value. Using a small positive value leads to an
    output tensor without NaN or Inf.

    Args:
    ----
        tensor (``torch.Tensor``): Specifies the input tensor.
        min_value (float, optional): Specifies the minimum value.
            Default: ``1e-8``

    Returns:
    -------
        ``torch.Tensor``: A new tensor with the natural logarithm
            of the elements.

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from gravitorch.utils.tensor import safelog
        >>> safelog(torch.tensor([1e-4, 1e-5, 1e-6, 1e-8, 1e-9, 1e-10]))
        tensor([ -9.2103, -11.5129, -13.8155, -18.4207, -18.4207, -18.4207])
    """
    return tensor.clamp(min=min_value).log()


def symlog(tensor: Tensor) -> Tensor:
    r"""Computes the symmetric logarithm natural logarithm of the
    elements.

    Note this transformation supports negative values.

    Args:
    ----
        tensor (``torch.Tensor``): Specifies the input tensor.

    Returns:
    -------
        ``torch.Tensor``: A new tensor with the symmetric natural
            logarithm of the elements.

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from gravitorch.utils.tensor import symlog
        >>> symlog(torch.tensor([-10.0, -1, 0.0, 1.0, 10.0]))
        tensor([-2.3979, -0.6931,  0.0000,  0.6931,  2.3979])
    """
    return tensor.sign().mul(tensor.abs().log1p())


def symlog_(tensor: Tensor) -> None:
    r"""Computes the symmetric logarithm natural of the elements.

    In-place version of ``symlog``.

    Note this transformation supports negative values.

    Args:
    ----
        tensor (``torch.Tensor``): Specifies the input tensor.

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from gravitorch.utils.tensor import symlog_
        >>> x = torch.tensor([-10.0, -1, 0.0, 1.0, 10.0])
        >>> symlog_(x)
        >>> x
        tensor([-2.3979, -0.6931,  0.0000,  0.6931,  2.3979])
    """
    sign = tensor.sign()
    tensor.abs_()
    tensor.log1p_()
    tensor.mul_(sign)


def isymlog(tensor: Tensor, max_value: float = 10.0) -> Tensor:
    r"""Computes the inverse of the symmetric logarithm natural of the
    elements.

    Args:
    ----
        tensor (``torch.Tensor``): Specifies the input tensor.
        max_value (float, optional): Specifies the maximum value.
            Default: ``10.0``

    Returns:
    -------
        ``torch.Tensor``: A new tensor with the inverse of the
            symmetric natural logarithm of the elements.

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from gravitorch.utils.tensor import isymlog
        >>> isymlog(torch.tensor([-2.0, -1, 0.0, 1.0, 2.0]))
        tensor([-6.3891, -1.7183,  0.0000,  1.7183,  6.3891])
    """
    sign = tensor.sign()
    return sign.mul(tensor.mul(sign).clamp(max=max_value).exp().sub(1))


def isymlog_(tensor: Tensor, max_value: float = 10.0) -> None:
    r"""Computes the inverse of the symmetric logarithm natural of the
    elements.

    In-place version of ``isymlog``.

    Args:
    ----
        tensor (``torch.Tensor``): Specifies the input tensor.
        max_value (float, optional): Specifies the maximum value.
            Default: ``10.0``

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from gravitorch.utils.tensor import isymlog_
        >>> x = torch.tensor([-2.0, -1, 0.0, 1.0, 2.0])
        >>> isymlog_(x)
        >>> x
        tensor([-6.3891, -1.7183,  0.0000,  1.7183,  6.3891])
    """
    sign = tensor.sign()
    tensor.mul_(sign)
    tensor.clamp_(max=max_value)
    tensor.exp_()
    tensor.sub_(1)
    tensor.mul_(sign)
