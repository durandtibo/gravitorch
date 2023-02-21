__all__ = ["FlattenModule", "MulticlassFlattenModule"]

from typing import Any, Union

import torch
from torch import Tensor
from torch.nn import Module

from gravitorch.nn.utils import setup_module


class FlattenModule(Module):
    r"""Implements a wrapper to flat the inputs of a ``torch.nn.Module``.

    Args:
        module (``torch.nn.Module`` or dict): Specifies the module or
            its configuration.
    """

    def __init__(self, module: Union[Module, dict]):
        super().__init__()
        self.module = setup_module(module)

    def forward(self, *args, **kwargs) -> Any:
        r"""Computes the forward pass of the module.

        All the inputs are converted to a 1d tensor.

        Args:
            *args: Same as ``module``
            **kwargs: Same as ``module``

        Returns:
            Same as ``module``
        """
        return self.module(
            *[torch.flatten(arg) for arg in args],
            **{key: torch.flatten(value) for key, value in kwargs.items()},
        )


class MulticlassFlattenModule(FlattenModule):
    r"""Implements a wrapper to flat the multiclass inputs of a
    ``torch.nn.Module``."""

    def forward(self, prediction: Tensor, target: Tensor) -> Any:
        r"""Computes the forward pass of the module.

        Args:
            prediction (``torch.Tensor`` of shape
                ``(batch_size, d1, d2, ..., dn, C)``): Specifies the
                predictions. This tensor will be reshaped to
                ``(batch_size * d1 * d2 * ... * dn, C)``.
            target (``torch.Tensor`` of shape
                ``(batch_size, d1, d2, ..., dn)``): Specifies the
                targets. This tensor will be reshaped to
                ``(batch_size * d1 * d2 * ... * dn,)``.

        Returns:
            Same as ``module``
        """
        return self.module(prediction.view(-1, prediction.shape[-1]), torch.flatten(target))
