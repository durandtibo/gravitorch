__all__ = ["ReLUn", "SquaredReLU"]

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F


class ReLUn(Module):
    r"""Implements the ReLU-n module.

    ``ReLUn(x, n)=min(max(0,x),n)``

    Args:
        max_value (float, optional): Specifies the maximum value.
            Default: ``1.0``
    """

    def __init__(self, max_value: float = 1.0):
        super().__init__()
        self._max_value = float(max_value)

    def extra_repr(self) -> str:
        return f"max_value={self._max_value}"

    def forward(self, tensor: Tensor) -> Tensor:
        r"""Applies the element-wise ReLU-n function.

        Args:
            tensor (``torch.Tensor`` of shape ``(*)``): Specifies the
                input tensor.

        Returns:
            ``torch.Tensor`` with same shape as the input: The output
                tensor.
        """
        return tensor.clamp(min=0.0, max=self._max_value)


class SquaredReLU(Module):
    r"""Implements the Squared ReLU.

    Squared ReLU is defined in the following paper:

        Primer: Searching for Efficient Transformers for Language Modeling.
        So DR., Mańke W., Liu H., Dai Z., Shazeer N., Le QV.
        NeurIPS, 2021. (https://arxiv.org/pdf/2109.08668.pdf)
    """

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        x = F.relu(tensor)
        return x * x
