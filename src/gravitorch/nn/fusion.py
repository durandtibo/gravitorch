r"""This module implements some concatenation based fusion layers."""

from __future__ import annotations

__all__ = [
    "AverageFusion",
    "ConcatFusion",
    "FusionFFN",
    "MultiplicationFusion",
    "SumFusion",
]


import torch
from torch import Tensor
from torch.nn import Module

from gravitorch.nn.utils import get_module_output_size, setup_module


class ConcatFusion(Module):
    r"""Implements a module to concatenate inputs.

    Args:
    ----
        dim (int, optional): Specifies the fusion dimension. ``-1``
            means the last dimension. Default: ``-1``

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from gravitorch.nn import ConcatFusion
        >>> module = ConcatFusion()
        >>> module
        ConcatFusion(dim=-1)
        >>> x1 = torch.tensor([[2, 3, 4], [5, 6, 7]], dtype=torch.float, requires_grad=True)
        >>> x2 = torch.tensor([[12, 13, 14], [15, 16, 17]], dtype=torch.float, requires_grad=True)
        >>> out = module(x1, x2)
        >>> out
        tensor([[ 2.,  3.,  4., 12., 13., 14.],
                [ 5.,  6.,  7., 15., 16., 17.]], grad_fn=<CatBackward0>)
        >>> out.mean().backward()
    """

    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self._dim = dim

    def extra_repr(self) -> str:
        return f"dim={self._dim}"

    def forward(self, *inputs: Tensor) -> Tensor:
        r"""Concatenates the list or tuple of inputs and then applied a
        feed-forward network (FFN) on the fused representation.

        Args:
        ----
            *inputs (list or tuple of ``torch.Tensor``): Specifies the
                tensors to concatenate.

        Returns:
        -------
            ``torch.Tensor``: The fused representation.
        """
        if not inputs:
            raise RuntimeError(f"{self.__class__.__qualname__} needs at least one tensor as input")
        return torch.cat(inputs, dim=self._dim)


class FusionFFN(Module):
    r"""Implements a module that fuses representations and then applies a
    feed-forward network (FFN) on the fused representation.

    Args:
    ----
        fusion (``torch.nn.Module``, optional): Specifies the fusion
            module or its configuration.
        ffn (``torch.nn.Module``, optional): Specifies the FFN or its
            configuration.

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from gravitorch.nn import ConcatFusion, FusionFFN
        >>> module = FusionFFN(ConcatFusion(), torch.nn.Linear(6, 4))
        >>> module
        FusionFFN(
          (fusion): ConcatFusion(dim=-1)
          (ffn): Linear(in_features=6, out_features=4, bias=True)
        )
        >>> x1 = torch.tensor([[2, 3, 4], [5, 6, 7]], dtype=torch.float, requires_grad=True)
        >>> x2 = torch.tensor([[12, 13, 14], [15, 16, 17]], dtype=torch.float, requires_grad=True)
        >>> out = module(x1, x2)
        >>> out
        tensor([[...]], grad_fn=<AddmmBackward0>)
        >>> out.mean().backward()
    """

    def __init__(self, fusion: Module | dict, ffn: Module | dict) -> None:
        super().__init__()
        self.fusion = setup_module(fusion)
        self.ffn = setup_module(ffn)

    @property
    def output_size(self) -> int:
        r"""``int``: The output size of the module."""
        return get_module_output_size(self.ffn)

    def forward(self, *inputs: Tensor) -> Tensor:
        r"""Fuses the inputs and then applied a feed-forward network
        (FFN) on the fused representation.

        Args:
        ----
            *inputs (sequence of ``torch.Tensor``): Specifies the
                sequence of tensors to fuse. The shape of the tensors
                may depend on the feed-forward network (FFN).

        Returns:
        -------
            ``torch.Tensor``: The fused representation.
        """
        return self.ffn(self.fusion(*inputs))


class MultiplicationFusion(Module):
    r"""Defines a fusion layer that multiplies the inputs.

    Example usage:

    .. code-block:: pycon

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
        ----
            *inputs (list or tuple of tensors): Specifies the list or
                tuple of tensors to multiply. The shape of the tensors
                should be the same. By default, this layer expects
                that each input is a ``torch.Tensor`` of shape
                ``(batch size, feature size)``. But it can also work
                if the inputs have a shape
                ``(sequence length, batch size, feature size)`` or
                similar shapes.

        Returns:
        -------
            ``torch.Tensor`` with the same shape that the input
                tensor: The fused tensor.
        """
        if not inputs:
            raise RuntimeError(f"{self.__class__.__qualname__} needs at least one tensor as input")
        output = inputs[0]
        for xi in inputs[1:]:
            output = output.mul(xi)
        return output


class SumFusion(Module):
    r"""Defines a layer to sum the inputs.

    Args:
    ----
        normalized (bool, optional): Specifies the output is
            normalized by the number of inputs.
            Default: ``False``

    Example usage:

    .. code-block:: pycon

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

    def __init__(self, normalized: bool = False) -> None:
        super().__init__()
        self._normalized = bool(normalized)

    def extra_repr(self) -> str:
        return f"normalized={self._normalized}"

    def forward(self, *inputs: Tensor) -> Tensor:
        r"""Sums the list or tuple of inputs.

        Args:
        ----
            *inputs (list or tuple of tensors): Specifies the list or
                tuple of tensors to sum. The shape of the tensors
                should be the same. By default, this layer expects
                that each input is a ``torch.Tensor`` of shape
                ``(batch size, feature size)``. But it can also work
                if the inputs have a shape
                ``(sequence length, batch size, feature size)`` or
                similar shapes.

        Returns:
        -------
            ``torch.Tensor`` with the same shape that the input
                tensor: The fused tensor.
        """
        if not inputs:
            raise RuntimeError(f"{self.__class__.__qualname__} needs at least one tensor as input")

        output = inputs[0]
        for x in inputs[1:]:
            output = output + x

        if self._normalized:
            output = output.div(len(inputs))
        return output


class AverageFusion(SumFusion):
    r"""Implements a layer to average the inputs.

    Example usage:

    .. code-block:: pycon

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

    def __init__(self) -> None:
        super().__init__(normalized=True)
