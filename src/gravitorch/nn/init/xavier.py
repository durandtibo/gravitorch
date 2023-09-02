from __future__ import annotations

__all__ = [
    "BaseXavier",
    "XavierNormal",
    "XavierUniform",
    "xavier_normal",
    "xavier_uniform",
]

import logging

from torch import nn
from torch.nn import Module

from gravitorch.nn.init.base import BaseInitializer

logger = logging.getLogger(__name__)


class BaseXavier(BaseInitializer):
    r"""Implements a module parameter initializer with the Xavier Normal
    or uniform strategy.

    Args:
    ----
        gain (float, optional): Specifies the gain or scaling factor.
            Default: ``1.0``
        learnable_only (bool, optional): If ``True``, only the
            learnable parameters are initialized, otherwise all the
            parameters are initialized. Default: ``True``
        log_info (bool, optional): If ``True``, log some information
            about the weights that are initialized. Default: ``False``
    """

    def __init__(
        self,
        gain: float = 1.0,
        learnable_only: bool = True,
        log_info: bool = False,
    ) -> None:
        super().__init__()
        self._gain = float(gain)
        self._learnable_only = bool(learnable_only)
        self._log_info = bool(log_info)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(gain={self._gain}, "
            f"learnable_only={self._learnable_only}, log_info={self._log_info})"
        )


class XavierNormal(BaseXavier):
    r"""Implements a module parameter initializer with the Xavier Normal
    strategy.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.nn.init import XavierNormal
        >>> from torch import nn
        >>> module = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.BatchNorm1d(6), nn.Linear(6, 1))
        >>> initializer = XavierNormal()
        >>> initializer
        XavierNormal(gain=1.0, learnable_only=True, log_info=False)
        >>> initializer.initialize(module)
    """

    def initialize(self, module: Module) -> None:
        logger.info(
            f"Initializing module parameters with the Xavier Normal strategy (gain={self._gain}, "
            f"learnable_only={self._learnable_only})..."
        )
        xavier_normal(
            module=module,
            gain=self._gain,
            learnable_only=self._learnable_only,
            log_info=self._log_info,
        )


class XavierUniform(BaseXavier):
    r"""Implements a module parameter initializer with the Xavier uniform
    strategy.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.nn.init import XavierUniform
        >>> from torch import nn
        >>> module = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.BatchNorm1d(6), nn.Linear(6, 1))
        >>> initializer = XavierUniform()
        >>> initializer
        XavierUniform(gain=1.0, learnable_only=True, log_info=False)
        >>> initializer.initialize(module)
    """

    def initialize(self, module: Module) -> None:
        logger.info(
            f"Initializing module parameters with the Xavier uniform strategy (gain={self._gain}, "
            f"learnable_only={self._learnable_only})..."
        )
        xavier_uniform(
            module=module,
            gain=self._gain,
            learnable_only=self._learnable_only,
            log_info=self._log_info,
        )


def xavier_normal(
    module: nn.Module,
    gain: float = 1.0,
    learnable_only: bool = True,
    log_info: bool = False,
) -> None:
    r"""Initialize the module parameters with the Xavier Normal strategy.

    Args:
    ----
        module (``torch.nn.Module``): Specifies the module to
            initialize.
        gain (float, optional): Specifies the gain or scaling factor.
            Default: ``1``
        learnable_only (bool, optional): If ``True``, only the
            learnable parameters are initialized, otherwise all the
            parameters are initialized. Default: ``True``
        log_info (bool, optional): If ``True``, log some information
            about the weights that are initialized. Default: ``False``

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.nn.init import xavier_normal
        >>> from torch import nn
        >>> module = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.BatchNorm1d(6), nn.Linear(6, 1))
        >>> xavier_normal(module)
    """
    for name, params in module.named_parameters():
        if params.ndim > 1 and (not learnable_only or learnable_only and params.requires_grad):
            if log_info:
                logger.info(f"Initializing '{name}' with gain {gain} | shape={params.shape}")
            nn.init.xavier_normal_(params.data, gain=gain)


def xavier_uniform(
    module: nn.Module,
    gain: float = 1.0,
    learnable_only: bool = True,
    log_info: bool = False,
) -> None:
    r"""Initializes the module parameters with the Xavier uniform
    strategy.

    Args:
    ----
        module (``torch.nn.Module``): Specifies the module to
            initialize.
        gain (float, optional): Specifies the gain or scaling factor.
            Default: ``1``
        learnable_only (bool, optional): If ``True``, only the
            learnable parameters are initialized, otherwise all the
            parameters are initialized. Default: ``True``
        log_info (bool, optional): If ``True``, log some information
            about the weights that are initialized. Default: ``False``

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.nn.init import xavier_uniform
        >>> from torch import nn
        >>> module = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.BatchNorm1d(6), nn.Linear(6, 1))
        >>> xavier_uniform(module)
    """
    for name, params in module.named_parameters():
        if params.ndim > 1 and (not learnable_only or learnable_only and params.requires_grad):
            if log_info:
                logger.info(f"Initializing '{name}' with gain {gain} | shape={params.shape}")
            nn.init.xavier_uniform_(params.data, gain=gain)
