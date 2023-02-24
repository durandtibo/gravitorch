__all__ = ["FusionNorm"]

from typing import Union

from torch import Tensor
from torch.nn import Dropout, LayerNorm, Module

from gravitorch.nn.fusion.multiplication import MultiplicationFusion
from gravitorch.nn.fusion.sum import SumFusion
from gravitorch.nn.utils import setup_module


class FusionNorm(Module):
    """Implements a layer that fuses multiple inputs and then computes a
    normalization.

    Args:
        fusion (``torch.nn.Module`` or dict): Specifies the fusion
            layer or its configuration.
        norm (``torch.nn.Module`` or dict): Specifies the
            normalization layer or its configuration.
        dropout (``torch.nn.Module`` or dict): Specifies the dropout
            layer or its configuration.

    Example usage:

    .. code-block:: python

        >>> import torch
        >>> from gravitorch.nn import FusionNorm
        >>> x1 = torch.tensor([[2, 3, 4], [5, 6, 7]], dtype=torch.float, requires_grad=True)
        >>> x2 = torch.tensor([[12, 13, 14], [15, 16, 17]], dtype=torch.float, requires_grad=True)
        >>> module = FusionNorm.create_sum_layer_norm(3)
        >>> module
        FusionNorm(
          (fusion): SumFusion(normalized=False)
          (norm): LayerNorm((3,), eps=1e-06, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        >>> out = module(x1, x2)
        >>> out
        tensor([[-1.3608,  0.0000,  1.3608],
                [-0.0000,  0.0000,  1.3608]], grad_fn=<MulBackward0>)
        >>> out.mean().backward()
        >>> module = FusionNorm.create_multiplication_layer_norm(3)
        >>> module
        FusionNorm(
          (fusion): MultiplicationFusion()
          (norm): LayerNorm((3,), eps=1e-06, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        >>> out = module(x1, x2)
        >>> out
        tensor([[-1.3316, -0.0567,  1.3883],
                [-1.3397, -0.0412,  1.3810]], grad_fn=<MulBackward0>)
        >>> out.mean().backward()
    """

    def __init__(
        self,
        fusion: Union[Module, dict],
        norm: Union[Module, dict],
        dropout: Union[Module, dict],
    ):
        super().__init__()
        self.fusion = setup_module(fusion)
        self.norm = setup_module(norm)
        self.dropout = setup_module(dropout)

    def forward(self, *inputs: Tensor) -> Tensor:
        r"""Fuses a list or tuple of inputs.

        Args:
            *inputs (list or tuple of tensors): Specifies the list or
                tuple of tensors to fuse. The shape of the tensors
                should be the same.

        Returns:
            ``torch.Tensor`` with the same shape that the input
                tensors: The fused tensor.
        """
        return self.dropout(self.norm(self.fusion(*inputs)))

    @classmethod
    def create_sum_layer_norm(
        cls,
        input_size: int,
        layer_norm_eps: float = 1e-6,
        dropout: float = 0.1,
    ) -> "FusionNorm":
        r"""Instantiates a ``FusionNorm`` layer with a sum fusion and a layer
        norm.

        Args:
            input_size (int): Specifies the input size.
            layer_norm_eps (float, optional): Specifies the ``eps``
                value in the ``LayerNorm`` layer. Default: ``1e-6``
            dropout (float, optional): Specifies the dropout value.
                Default: ``0.1``

        Returns:
            ``FusionNorm``: A ``FusionNorm`` layer with a sum fusion
                and a layer norm.
        """
        return cls(
            fusion=SumFusion(),
            norm=LayerNorm(input_size, eps=layer_norm_eps),
            dropout=Dropout(dropout),
        )

    @classmethod
    def create_multiplication_layer_norm(
        cls,
        input_size: int,
        layer_norm_eps: float = 1e-6,
        dropout: float = 0.1,
    ) -> "FusionNorm":
        r"""Instantiates a ``FusionNorm`` layer with a multiplication fusion and
        a layer norm.

        Args:
            input_size (int): Specifies the input size.
            layer_norm_eps (float, optional): Specifies the ``eps``
                value in the ``LayerNorm`` layer. Default: ``1e-6``
            dropout (float, optional): Specifies the dropout value.
                Default: ``0.1``

        Returns:
            ``FusionNorm``: A ``FusionNorm`` layer with a
                multiplication fusion and a layer norm.
        """
        return cls(
            fusion=MultiplicationFusion(),
            norm=LayerNorm(input_size, eps=layer_norm_eps),
            dropout=Dropout(dropout),
        )
