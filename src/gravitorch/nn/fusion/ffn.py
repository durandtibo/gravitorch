__all__ = ["FusionFFN"]

from typing import Union

from torch import Tensor
from torch.nn import Module

from gravitorch.nn.utils import get_module_output_size, setup_module


class FusionFFN(Module):
    r"""Implements a module that fuses representations and then applies a feed-
    forward network (FFN) on the fused representation.

    Args:
        fusion (``torch.nn.Module``, optional): Specifies the fusion
            module or its configuration.
        ffn (``torch.nn.Module``, optional): Specifies the FFN or its
            configuration.

    Example usage:

    .. code-block:: python

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
        tensor([[ 2.1302,  7.8643,  8.3265,  4.7277],
                [ 1.6980, 11.7446,  9.8449,  5.8402]], grad_fn=<AddmmBackward0>)
        >>> out.mean().backward()
    """

    def __init__(self, fusion: Union[Module, dict], ffn: Union[Module, dict]):
        super().__init__()
        self.fusion = setup_module(fusion)
        self.ffn = setup_module(ffn)

    @property
    def output_size(self) -> int:
        r"""int: The output size of the module."""
        return get_module_output_size(self.ffn)

    def forward(self, *inputs: Tensor) -> Tensor:
        r"""Fuses the inputs and then applied a feed-forward network (FFN) on
        the fused representation.

        Args:
            *inputs (sequence of ``torch.Tensor``): Specifies the
                sequence of tensors to fuse. The shape of the tensors
                may depend on the feed-forward network (FFN).

        Returns:
            ``torch.Tensor``: The fused representation.
        """
        return self.ffn(self.fusion(*inputs))
