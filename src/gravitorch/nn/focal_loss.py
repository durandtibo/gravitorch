r"""This module contains several focal loss implementations."""

__all__ = ["BinaryFocalLoss", "BinaryFocalLossWithLogits"]

import torch
from torch import Tensor
from torch.nn import BCELoss, BCEWithLogitsLoss, Module

from gravitorch.nn.functional import basic_loss_reduction, check_basic_loss_reduction


class _BinaryFocalLoss(Module):
    r"""Base class to implement the binary Focal Loss with and without logits.

    Based on "Focal Loss for Dense Object Detection"
    (https://arxiv.org/pdf/1708.02002.pdf)

    Args:
        alpha (float, optional): Specifies the weighting factor in
            ``[0, 1]``.
        gamma (float, optional): Specifies the focusing parameter
            (``>=0``).
        reduction (string, optional): Specifies the reduction to
            apply to the output: ``'none'`` | ``'mean'`` | ``'sum'``.
            ``'none'``: no reduction will be applied, ``'mean'``: the
            sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be
            summed.
    """

    def __init__(
        self,
        bce: Module,
        alpha: float,
        gamma: float,
        reduction: str,
    ):
        super().__init__()
        self.bce = bce

        if 0 <= alpha <= 1:
            self._alpha = float(alpha)
        else:
            raise ValueError(
                f"Incorrect parameter alpha ({alpha}). The valid range of value is [0, 1]."
            )

        if gamma >= 0:
            self._gamma = float(gamma)
        else:
            raise ValueError(
                f"Incorrect parameter gamma ({gamma}). Gamma has to be positive (>=0)."
            )

        check_basic_loss_reduction(reduction)
        self.reduction = reduction

    def extra_repr(self) -> str:
        return f"alpha={self._alpha}, gamma={self._gamma}, reduction={self.reduction}"

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        r"""Computes the binary Focal Loss.

        Args:
            prediction (``torch.Tensor`` of shape
                ``(batch size, num classes)`` and type float):
                Specifies the predicted probabilities. All elements
                should be between ``0`` and ``1``.
            target (``torch.Tensor`` of shape
                ``(batch size, num classes)`` and type float):
                Specifies the targets. ``1`` (resp. ``0``) means a
                positive (resp. negative) example.

        Returns:
            ``torch.Tensor`` of type float: The loss value(s). The
                shape of the tensor depends on the reduction. If the
                reduction is ``mean`` or ``sum``, the tensor has a
                single scalar value. If the reduction is ``none``,
                the shape of the tensor is the same that the inputs.
        """
        bce_loss = self.bce(prediction, target)
        pt = torch.exp(-bce_loss)
        alpha_t = self._alpha * target + (1 - self._alpha) * (
            1 - target
        )  # alpha for positive samples, else 1-alpha
        focal_loss = alpha_t * (1 - pt) ** self._gamma * bce_loss
        return basic_loss_reduction(focal_loss, self.reduction)


class BinaryFocalLoss(_BinaryFocalLoss):
    r"""Implementation of the binary Focal Loss.

    Based on "Focal Loss for Dense Object Detection"
    (https://arxiv.org/pdf/1708.02002.pdf)

    Args:
        alpha (float, optional): Specifies the weighting factor in
            ``[0, 1]``. Default: ``0.5``
        gamma (float, optional): Specifies the focusing parameter
            (``>=0``). Default: ``2``
        reduction (string, optional): Specifies the reduction to
            apply to the output: ``'none'`` | ``'mean'`` | ``'sum'``.
            ``'none'``: no reduction will be applied, ``'mean'``: the
            sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be
            summed. Default: ``"mean"``
    """

    def __init__(
        self,
        alpha: float = 0.5,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__(
            bce=BCELoss(reduction="none"),
            alpha=alpha,
            gamma=gamma,
            reduction=reduction,
        )


class BinaryFocalLossWithLogits(_BinaryFocalLoss):
    r"""Implementation of the binary Focal Loss with logits.

    Based on "Focal Loss for Dense Object Detection"
    (https://arxiv.org/pdf/1708.02002.pdf)

    The predictions do not have to be between ``0`` and ``1`` because
    they will be converted to probability.

    Args:
        alpha (float, optional): Specifies the weighting factor in
            ``[0, 1]``. Default: ``0.5``
        gamma (float, optional): Specifies the focusing parameter
            (``>=0``). Default: ``2``
        reduction (string, optional): Specifies the reduction to
            apply to the output: ``'none'`` | ``'mean'`` | ``'sum'``.
            ``'none'``: no reduction will be applied, ``'mean'``: the
            sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be
            summed. Default: ``"mean"``
    """

    def __init__(
        self,
        alpha: float = 0.5,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__(
            bce=BCEWithLogitsLoss(reduction="none"),
            alpha=alpha,
            gamma=gamma,
            reduction=reduction,
        )
