__all__ = ["ExU"]

from typing import Union

import torch
from torch import Tensor
from torch.nn import Module, Parameter, ReLU
from torch.nn.init import trunc_normal_, zeros_

from gravitorch.nn.utils import setup_module


class ExU(Module):
    r"""Implementation of the exp-centered (ExU) layer.

    This layer was proposed in the following paper:

        Neural Additive Models: Interpretable Machine Learning with
        Neural Nets.
        Agarwal R., Melnick L., Frosst N., Zhang X., Lengerich B.,
        Caruana R., Hinton G.
        NeurIPS 2021. (https://arxiv.org/pdf/2004.13912.pdf)

    Args:
        input_size (int): Specifies the input size.
        output_size (int): Specifies the output size.
        activation (``torch.nn.Module`` or dict or ``None``):
            Specifies the activation layer or its configuration.
            If ``None``, the ReLU layer is used. Default: ``None``
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation: Union[Module, dict, None] = None,
    ):
        super().__init__()
        self.weight = Parameter(torch.empty(input_size, output_size))
        self.bias = Parameter(torch.zeros(input_size))
        self.activation = setup_module(activation or ReLU())

        self.reset_parameters()

    @property
    def input_size(self) -> int:
        r"""int: The input size"""
        return self.weight.shape[0]

    @property
    def output_size(self) -> int:
        r"""int: The output size"""
        return self.weight.shape[1]

    def extra_repr(self) -> str:
        return f"input_size={self.input_size}, output_size={self.output_size}"

    def forward(self, tensor: Tensor) -> Tensor:
        r"""Computes the forward of the ExU layer.

        Args:
            tensor (``torch.Tensor`` of type float and shape
                ``(d0, d1, ..., dn, input_size)``): Specifies the
                input tensor.

        Returns:
            ``torch.Tensor`` of type float and shape
                ``(d0, d1, ..., dn, output_size)``: The output
                tensor.
        """
        return self.activation(tensor.sub(self.bias).matmul(self.weight.exp()))

    def reset_parameters(self) -> None:
        r"""Resets the parameters.

        As indicated in page 4 of the paper, the weights are initialed
        using a normal distribution ``N(4.0; 0.5)``. The biases are
        initialized to ``0``
        """
        mean, std = 4.0, 0.5
        trunc_normal_(self.weight, mean=mean, std=std, a=mean - 3 * std, b=mean + 3 * std)
        zeros_(self.bias)
