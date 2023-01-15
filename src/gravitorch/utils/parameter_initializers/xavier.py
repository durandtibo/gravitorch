__all__ = [
    "BaseXavierParameterInitializer",
    "XavierNormalParameterInitializer",
    "XavierUniformParameterInitializer",
    "recursive_xavier_normal_",
    "recursive_xavier_uniform_",
]

import logging

from torch import nn

from gravitorch.engines.base import BaseEngine
from gravitorch.utils.parameter_initializers.base import BaseDefaultParameterInitializer

logger = logging.getLogger(__name__)


class BaseXavierParameterInitializer(BaseDefaultParameterInitializer):
    r"""Implements a model parameter initializer with the Xavier Normal or
    uniform strategy.

    Args:
        gain (float, optional): Specifies the gain or scaling factor.
            Default: ``1``
        learnable_only (bool, optional): If ``True``, only the
            learnable parameters are initialized, otherwise all the
            parameters are initialized. Default: ``True``
        show_stats (bool, optional): If ``True``, the parameter
            statistics are shown at the end of the initialization.
            Default: ``True``
    """

    def __init__(self, gain: float = 1.0, learnable_only: bool = True, show_stats: bool = True):
        super().__init__(show_stats=show_stats)
        self._gain = float(gain)
        self._learnable_only = bool(learnable_only)

    def __str__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(gain={self._gain}, "
            f"learnable_only={self._learnable_only})"
        )


class XavierNormalParameterInitializer(BaseXavierParameterInitializer):
    r"""Implements a model parameter initializer with the Xavier Normal
    strategy."""

    def _initialize(self, engine: BaseEngine) -> None:
        logger.info("Initializing model parameters with the Xavier Normal strategy...")
        recursive_xavier_normal_(
            module=engine.model, gain=self._gain, learnable_only=self._learnable_only
        )


class XavierUniformParameterInitializer(BaseXavierParameterInitializer):
    r"""Implements a model parameter initializer with the Xavier uniform
    strategy."""

    def _initialize(self, engine: BaseEngine) -> None:
        logger.info("Initializing model parameters with the Xavier uniform strategy...")
        recursive_xavier_uniform_(
            module=engine.model, gain=self._gain, learnable_only=self._learnable_only
        )


def recursive_xavier_normal_(
    module: nn.Module, gain: float = 1.0, learnable_only: bool = True
) -> None:
    r"""Initialize the parameters of the module with the Xavier Normal
    initialization.

    Args:
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

        >>> from gravitorch.utils.parameter_initializers import recursive_xavier_normal_
        >>> from torch import nn
        >>> net = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.BatchNorm1d(6), nn.Linear(6, 1))
        >>> recursive_xavier_normal_(net)
    """
    for params in module.parameters():
        if params.ndim > 1 and (not learnable_only or learnable_only and params.requires_grad):
            nn.init.xavier_normal_(params.data, gain=gain)


def recursive_xavier_uniform_(
    module: nn.Module, gain: float = 1.0, learnable_only: bool = True
) -> None:
    r"""Initialize the parameters of the module with the Xavier uniform
    initialization.

    Args:
        module (``torch.nn.Module``): Specifies the module to
            initialize.
        gain (float, optional): Specifies the gain or scaling factor.
            Default: ``1``
        learnable_only (bool, optional): If ``True``, only the
            learnable parameters are initialized, otherwise all the
            parameters are initialized. Default: ``True``

    Example usage:

    .. code-block:: python

        >>> from gravitorch.utils.parameter_initializers import recursive_xavier_uniform_
        >>> from torch import nn
        >>> net = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.BatchNorm1d(6), nn.Linear(6, 1))
        >>> recursive_xavier_uniform_(net)
    """
    for params in module.parameters():
        if params.ndim > 1 and (not learnable_only or learnable_only and params.requires_grad):
            nn.init.xavier_uniform_(params.data, gain=gain)
