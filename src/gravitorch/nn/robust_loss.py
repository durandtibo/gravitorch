__all__ = [
    "AsinhMSELoss",
    "MSLELoss",
    "RelativeMSELoss",
    "RelativeSmoothL1Loss",
    "SymlogMSELoss",
    "SymmetricRelativeSmoothL1Loss",
]

from torch import Tensor
from torch.nn import Module

from gravitorch.nn.functional import (
    asinh_mse_loss,
    msle_loss,
    relative_mse_loss,
    relative_smooth_l1_loss,
    symlog_mse_loss,
    symmetric_relative_smooth_l1_loss,
)


class AsinhMSELoss(Module):
    r"""Implements the mean squared error (MSE) loss with inverse hyperbolic
    sine (arcsinh) transformation.

    Args:
        reduction (string, optional): Specifies the reduction to apply
            to the output: ``'none'`` | ``'mean'`` | ``'sum'``.
            ``'none'``: no reduction will be applied, ``'mean'``: the
            sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be
            summed. Default: ``'mean'``

    Example usage:

    .. code-block:: python

        >>> import torch
        >>> from gravitorch.nn import AsinhMSELoss
        >>> criterion = AsinhMSELoss()
        >>> loss = criterion(torch.randn(3, 5, requires_grad=True), torch.randn(3, 5))
        >>> loss.backward()
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = str(reduction)

    def extra_repr(self) -> str:
        return f"reduction={self.reduction}"

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return asinh_mse_loss(input, target, self.reduction)


class MSLELoss(Module):
    r"""Implements the mean squared logarithmic error (MSLE) loss.

    Note: this loss only works with positive values (0 included).

    Args:
        reduction (string, optional): Specifies the reduction to apply
            to the output: ``'none'`` | ``'mean'`` | ``'sum'``.
            ``'none'``: no reduction will be applied, ``'mean'``: the
            sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be
            summed. Default: ``'mean'``

    Example usage:

    .. code-block:: python

        >>> import torch
        >>> from gravitorch.nn import MSLELoss
        >>> criterion = MSLELoss()
        >>> loss = criterion(torch.rand(3, 5, requires_grad=True), torch.rand(3, 5))
        >>> loss.backward()
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = str(reduction)

    def extra_repr(self) -> str:
        return f"reduction={self.reduction}"

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return msle_loss(input, target, self.reduction)


class SymlogMSELoss(Module):
    r"""Implements the mean squared error (MSE) loss with symmetric logarithmic
    (symlog) transformation.

    Args:
        reduction (string, optional): Specifies the reduction to apply
            to the output: ``'none'`` | ``'mean'`` | ``'sum'``.
            ``'none'``: no reduction will be applied, ``'mean'``: the
            sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be
            summed. Default: ``'mean'``

    Example usage:

    .. code-block:: python

        >>> import torch
        >>> from gravitorch.nn import SymlogMSELoss
        >>> criterion = SymlogMSELoss()
        >>> loss = criterion(torch.randn(3, 5, requires_grad=True), torch.randn(3, 5))
        >>> loss.backward()
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = str(reduction)

    def extra_repr(self) -> str:
        return f"reduction={self.reduction}"

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return symlog_mse_loss(input, target, self.reduction)


class RelativeMSELoss(Module):
    r"""Implements the relative mean squared error.

    Args:
        reduction (string, optional): Specifies the reduction to apply
            to the output: ``'none'`` | ``'mean'`` | ``'sum'``.
            ``'none'``: no reduction will be applied, ``'mean'``: the
            sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be
            summed. Default: ``'mean'``
        eps (float, optional): Specifies an arbitrary small strictly
            positive number to avoid undefined results when the target
            is zero. Default: ``1e-8``

    Example usage:

    .. code-block:: python

        >>> import torch
        >>> from gravitorch.nn import RelativeMSELoss
        >>> criterion = RelativeMSELoss()
        >>> loss = criterion(torch.randn(3, 5, requires_grad=True), torch.randn(3, 5))
        >>> loss.backward()
    """

    def __init__(self, reduction: str = "mean", eps: float = 1e-8) -> None:
        super().__init__()
        self.reduction = str(reduction)
        self._eps = float(eps)

    def extra_repr(self) -> str:
        return f"reduction={self.reduction}"

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return relative_mse_loss(input, target, reduction=self.reduction, eps=self._eps)


class RelativeSmoothL1Loss(Module):
    r"""Implements the relative smooth L1 loss.

    Args:
        reduction (string, optional): Specifies the reduction to apply
            to the output: ``'none'`` | ``'mean'`` | ``'sum'``.
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

    Example usage:

    .. code-block:: python

        >>> import torch
        >>> from gravitorch.nn import RelativeSmoothL1Loss
        >>> criterion = RelativeSmoothL1Loss()
        >>> loss = criterion(torch.randn(3, 5, requires_grad=True), torch.randn(3, 5))
        >>> loss.backward()
    """

    def __init__(self, reduction: str = "mean", beta: float = 1.0, eps: float = 1e-8):
        super().__init__()
        self.reduction = str(reduction)
        self._beta = float(beta)
        self._eps = float(eps)

    def extra_repr(self) -> str:
        return f"reduction={self.reduction}, beta={self._beta}, eps={self._eps}"

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return relative_smooth_l1_loss(
            input,
            target,
            reduction=self.reduction,
            beta=self._beta,
            eps=self._eps,
        )


class SymmetricRelativeSmoothL1Loss(Module):
    r"""Implements the symmetric relative smooth L1 loss.

    Args:
        reduction (string, optional): Specifies the reduction to apply
            to the output: ``'none'`` | ``'mean'`` | ``'sum'``.
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

    Example usage:

    .. code-block:: python

        >>> import torch
        >>> from gravitorch.nn import SymmetricRelativeSmoothL1Loss
        >>> criterion = SymmetricRelativeSmoothL1Loss()
        >>> loss = criterion(torch.randn(3, 5, requires_grad=True), torch.randn(3, 5))
        >>> loss.backward()
    """

    def __init__(self, reduction: str = "mean", beta: float = 1.0, eps: float = 1e-8):
        super().__init__()
        self.reduction = str(reduction)
        self._beta = float(beta)
        self._eps = float(eps)

    def extra_repr(self) -> str:
        return f"reduction={self.reduction}, beta={self._beta}, eps={self._eps}"

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return symmetric_relative_smooth_l1_loss(
            input,
            target,
            reduction=self.reduction,
            beta=self._beta,
            eps=self._eps,
        )
