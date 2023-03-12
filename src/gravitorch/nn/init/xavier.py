__all__ = [
    "BaseXavier",
    "XavierNormal",
    "XavierUniform",
    "xavier_normal_init",
    "xavier_uniform_init",
]

import logging

from torch import nn
from torch.nn import Module

from gravitorch.nn.init.base import BaseInitializer

logger = logging.getLogger(__name__)


class BaseXavier(BaseInitializer):
    r"""Implements a model parameter initializer with the Xavier Normal or
    uniform strategy.

    Args:
    ----
        gain (float, optional): Specifies the gain or scaling factor.
            Default: ``1``
        learnable_only (bool, optional): If ``True``, only the
            learnable parameters are initialized, otherwise all the
            parameters are initialized. Default: ``True``
    """

    def __init__(self, gain: float = 1.0, learnable_only: bool = True) -> None:
        super().__init__()
        self._gain = float(gain)
        self._learnable_only = bool(learnable_only)

    def __str__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(gain={self._gain}, "
            f"learnable_only={self._learnable_only})"
        )


class XavierNormal(BaseXavier):
    r"""Implements a model parameter initializer with the Xavier Normal
    strategy."""

    def initialize(self, module: Module) -> None:
        logger.info("Initializing module parameters with the Xavier Normal strategy...")
        xavier_normal_init(module=module, gain=self._gain, learnable_only=self._learnable_only)


class XavierUniform(BaseXavier):
    r"""Implements a model parameter initializer with the Xavier uniform
    strategy."""

    def initialize(self, module: Module) -> None:
        logger.info("Initializing module parameters with the Xavier uniform strategy...")
        xavier_uniform_init(module=module, gain=self._gain, learnable_only=self._learnable_only)


def xavier_normal_init(module: nn.Module, gain: float = 1.0, learnable_only: bool = True) -> None:
    r"""Initialize the parameters of the module with the Xavier Normal
    initialization.

    Args:
    ----
        module (``torch.nn.Module``): Specifies the module to
            initialize.
        gain (float, optional): Specifies the gain or scaling factor.
            Default: ``1``
        learnable_only (bool, optional): If ``True``, only the
            learnable parameters are initialized,
            otherwise all the parameters are initialized.
            Default: ``True``

    Example usage:

    .. code-block:: python

        >>> from gravitorch.nn.init import xavier_normal_init
        >>> from torch import nn
        >>> net = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.BatchNorm1d(6), nn.Linear(6, 1))
        >>> xavier_normal_init(net)
    """
    for params in module.parameters():
        if params.ndim > 1 and (not learnable_only or learnable_only and params.requires_grad):
            nn.init.xavier_normal_(params.data, gain=gain)


def xavier_uniform_init(module: nn.Module, gain: float = 1.0, learnable_only: bool = True) -> None:
    r"""Initialize the parameters of the module with the Xavier uniform
    initialization.

    Args:
    ----
        module (``torch.nn.Module``): Specifies the module to
            initialize.
        gain (float, optional): Specifies the gain or scaling factor.
            Default: ``1``
        learnable_only (bool, optional): If ``True``, only the
            learnable parameters are initialized, otherwise all the
            parameters are initialized. Default: ``True``

    Example usage:

    .. code-block:: python

        >>> from gravitorch.nn.init import xavier_uniform_init
        >>> from torch import nn
        >>> net = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.BatchNorm1d(6), nn.Linear(6, 1))
        >>> xavier_uniform_init(net)
    """
    for params in module.parameters():
        if params.ndim > 1 and (not learnable_only or learnable_only and params.requires_grad):
            nn.init.xavier_uniform_(params.data, gain=gain)
