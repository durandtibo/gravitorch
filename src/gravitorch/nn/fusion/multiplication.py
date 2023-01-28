r"""This module implements some multiplication-like fusion layers."""

__all__ = ["MultiplicationFusion"]

from torch import Tensor
from torch.nn import Module


class MultiplicationFusion(Module):
    r"""Defines a fusion layer that multiplies the inputs.

    Example usage:

    .. code-block:: python

        >>> import torch
        >>> from gravitorch.nn import MultiplicationFusion
        >>> module = MultiplicationFusion()
        >>> module
        MultiplicationFusion()
        >>> x1 = torch.tensor([[2, 3, 4], [5, 6, 7]], dtype=torch.float, requires_grad=True)
        >>> x2 = torch.tensor([[12, 13, 14], [15, 16, 17]], dtype=torch.float, requires_grad=True)
        >>> out = module(x1, x2)
        >>> out
        tensor([[ 24.,  39.,  56.],
                [ 75.,  96., 119.]], grad_fn=<MulBackward0>)
        >>> out.mean().backward()
    """

    def forward(self, *inputs: Tensor) -> Tensor:
        r"""Multiplies the list or tuple of inputs.

        Args:
            *inputs (list or tuple of tensors): Specifies the list or
                tuple of tensors to multiply. The shape of the tensors
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
        if len(inputs) == 0:
            raise ValueError(
                f"{type(self).__qualname__} must have at least one tensor in the input"
            )
        output = inputs[0]
        for xi in inputs[1:]:
            output = output.mul(xi)
        return output
