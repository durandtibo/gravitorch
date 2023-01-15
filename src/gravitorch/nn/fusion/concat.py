r"""This module implements some concatenation based fusion layers."""

__all__ = ["ConcatFusion"]

import torch
from torch import Tensor
from torch.nn import Module


class ConcatFusion(Module):
    r"""Implements a module to concatenate inputs.

    Args:
        dim (int, optional): Specifies the fusion dimension. ``-1``
            means the last dimension. Default: ``-1``

    Example usage:

    .. code-block:: python

        >>> import torch
        >>> from gravitorch.nn import ConcatFusion
        >>> module = ConcatFusion()
        >>> module
        ConcatFusion(dim=-1)
        >>> x1 = torch.tensor([[2, 3, 4], [5, 6, 7]], dtype=torch.float, requires_grad=True)
        >>> x2 = torch.tensor([[12, 13, 14], [15, 16, 17]], dtype=torch.float, requires_grad=True)
        >>> out = module(x1, x2)
        tensor([[ 2.,  3.,  4., 12., 13., 14.],
                [ 5.,  6.,  7., 15., 16., 17.]], grad_fn=<CatBackward0>)
        >>> out.mean().backward()
    """

    def __init__(self, dim: int = -1):
        super().__init__()
        self._dim = dim

    def extra_repr(self) -> str:
        return f"dim={self._dim}"

    def forward(self, *inputs: Tensor) -> Tensor:
        r"""Concatenates the list or tuple of inputs and then applied a feed-
        forward network (FFN) on the fused representation.

        Args:
            *inputs (list or tuple of ``torch.Tensor``): Specifies the
                tensors to concatenate.

        Returns:
            ``torch.Tensor``: The fused representation.
        """
        return torch.cat(inputs, dim=self._dim)
