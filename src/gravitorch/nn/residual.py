__all__ = ["ResidualBlock"]

from typing import Union

from torch import Tensor
from torch.nn import Identity, Module

from gravitorch.nn.utils import setup_module


class ResidualBlock(Module):
    r"""Implementation of a residual block.

    Args:
        residual (``torch.nn.Module`` or dict): Specifies the residual
            mapping module or its configuration (dictionary).
        skip (``torch.nn.Module`` or dict or ``None``): Specifies the
            skip mapping module or its configuration (dictionary). If
            ``None``, the ``Identity`` module is used.
            Default: ``None``
    """

    def __init__(
        self,
        residual: Union[Module, dict],
        skip: Union[Module, dict, None] = None,
    ):
        super().__init__()
        self.residual = setup_module(residual)
        self.skip = setup_module(skip or Identity())

    def forward(self, tensor: Tensor) -> Tensor:
        return self.skip(tensor) + self.residual(tensor)
