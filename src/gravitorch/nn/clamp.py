__all__ = ["Clamp", "ClampLoss"]

from typing import Optional, Union

from torch import Tensor
from torch.nn import Module

from gravitorch.nn.functional import basic_loss_reduction, check_basic_loss_reduction
from gravitorch.nn.utils import setup_module


class Clamp(Module):
    r"""Implements a layer to clamp the tensor values.

    Args:
        min_value (float or ``None``, optional): Specifies the
            minimum value. ``None`` means there is no minimum
            value. Default: ``-1.0``
        max_value (float or ``None``, optional): Specifies the
            maximum value. ``None`` means there is no maximum
            value. Default: ``1.0``
    """

    def __init__(self, min_value: Optional[float] = -1.0, max_value: Optional[float] = 1.0):
        super().__init__()
        self._min_value = min_value
        self._max_value = max_value

    def extra_repr(self) -> str:
        return f"min_value={self._min_value}, max_value={self._max_value}"

    def forward(self, tensor: Tensor) -> Tensor:
        r"""Applies the element-wise ReLU-n function.

        Args:
            tensor (``torch.Tensor`` of shape ``(*)``): Specifies the
                input tensor.

        Returns:
            ``torch.Tensor`` with same shape as the input: The output
                tensor.
        """
        return tensor.clamp(min=self._min_value, max=self._max_value)


class ClampLoss(Module):
    r"""Implements a loss function wrapper that clamps output values.

    Args:
        criterion (``torch.nn.Module`` or dict): Specifies the loss
            function to wrap or its configuration.
        min_value (float or ``None``, optional): Specifies the
            minimum value. ``None`` means there is no minimum
            value. Default: ``-1.0``
        max_value (float or ``None``, optional): Specifies the
            maximum value. ``None`` means there is no maximum
            value. Default: ``1.0``
        reduction (string, optional): Specifies the reduction to apply
            to the output: ``'none'`` | ``'mean'`` | ``'sum'``.
            ``'none'``: no reduction will be applied, ``'mean'``: the
            sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be
            summed. Default: ``'mean'``
    """

    def __init__(
        self,
        criterion: Union[Module, dict],
        min_value: Optional[float],
        max_value: Optional[float],
        reduction: str = "none",
    ):
        super().__init__()
        self.criterion = setup_module(criterion)
        self.clamp = Clamp(min_value=min_value, max_value=max_value)

        check_basic_loss_reduction(reduction)
        self.reduction = reduction

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        r"""Computes the loss values, then clamps the values and reduces them.

        Args:
            prediction (``torch.Tensor``): Specifies the predictions.
            target (``torch.Tensor``): Specifies the targets.

        Returns:
            ``torch.Tensor``: The computed loss value.
        """
        return basic_loss_reduction(self.clamp(self.criterion(prediction, target)), self.reduction)
