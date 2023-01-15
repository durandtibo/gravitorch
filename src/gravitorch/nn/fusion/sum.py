r"""This module implements some sum-like fusion layers."""

__all__ = ["SumFusion", "AverageFusion"]

from torch import Tensor
from torch.nn import Module


class SumFusion(Module):
    r"""Defines a layer to sum the inputs.

    Args:
        normalized (bool, optional): Specifies the output is
            normalized by the number of inputs.
            Default: ``False``

    Example usage:

    .. code-block:: python

        >>> import torch
        >>> from gravitorch.nn import SumFusion
        >>> module = SumFusion()
        >>> module
        SumFusion(normalized=False)
        >>> x1 = torch.tensor([[2, 3, 4], [5, 6, 7]], dtype=torch.float, requires_grad=True)
        >>> x2 = torch.tensor([[12, 13, 14], [15, 16, 17]], dtype=torch.float, requires_grad=True)
        >>> out = module(x1, x2)
        >>> out
        tensor([[14., 16., 18.],
                [20., 22., 24.]], grad_fn=<AddBackward0>)
        >>> out.mean().backward()
    """

    def __init__(self, normalized: bool = False):
        super().__init__()
        self._normalized = normalized

    def extra_repr(self) -> str:
        return f"normalized={self._normalized}"

    def forward(self, *inputs: Tensor) -> Tensor:
        r"""Sums the list or tuple of inputs.

        Args:
            *inputs (list or tuple of tensors): Specifies the list or
                tuple of tensors to sum. The shape of the tensors
                should be the same. By default, this layer expects
                that each input is a ``torch.Tensor`` of shape
                ``(batch size, feature size)``. But it can also work
                if the inputs have a shape
                ``(sequence length, batch size, feature size)`` or
                similar shapes.

        Returns:
            ``torch.Tensor`` with the same shape that the input
                tensor: The fused tensor.
        """
        if not inputs:
            raise ValueError("This fusion needs to have at least one tensor in the input")

        output = inputs[0]
        for x in inputs[1:]:
            output = output + x

        if self._normalized:
            output = output.div(len(inputs))
        return output


class AverageFusion(SumFusion):
    r"""Implements a layer to average the inputs.

    Example usage:

    .. code-block:: python

        >>> import torch
        >>> from gravitorch.nn import AverageFusion
        >>> module = AverageFusion()
        >>> module
        AverageFusion(normalized=True)
        >>> x1 = torch.tensor([[2, 3, 4], [5, 6, 7]], dtype=torch.float, requires_grad=True)
        >>> x2 = torch.tensor([[12, 13, 14], [15, 16, 17]], dtype=torch.float, requires_grad=True)
        >>> out = module(x1, x2)
        >>> out
        tensor([[ 7.,  8.,  9.],
                [10., 11., 12.]], grad_fn=<DivBackward0>)
        >>> out.mean().backward()
    """

    def __init__(self):
        super().__init__(normalized=True)
