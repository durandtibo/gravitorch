__all__ = ["ReLUn"]

from torch import Tensor
from torch.nn import Module


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
