from __future__ import annotations

__all__ = [
    "BaseKaiming",
    "KaimingNormal",
    "KaimingUniform",
    "kaiming_normal",
    "kaiming_uniform",
]

import logging

from torch import nn
from torch.nn import Module

from gravitorch.nn.init.base import BaseInitializer

logger = logging.getLogger(__name__)


class BaseKaiming(BaseInitializer):
    r"""Implements a module parameter initializer with the Kaiming Normal
    or uniform strategy.

    Args:
    ----
        neg_slope (float, optional): Specifies the negative slope of
            the rectifier used after this layer (only used with
            ``'leaky_relu'``). Default: ``0.0``
        mode (str, optional): either ``'fan_in'`` or ``'fan_out'``.
            Choosing ``'fan_in'`` preserves the magnitude of the
            variance of the weights in the forward pass. Choosing
            ``'fan_out'`` preserves the magnitudes in the backwards
            pass. Default: ``'fan_in'``
        nonlinearity (str, optional): the non-linear function
            (`nn.functional` name), recommended to use only with
            ``'relu'`` or ``'leaky_relu'``. Default: ``'leaky_relu'``
        learnable_only (bool, optional): If ``True``, only the
            learnable parameters are initialized, otherwise all the
            parameters are initialized. Default: ``True``
        log_info (bool, optional): If ``True``, log some information
            about the weights that are initialized. Default: ``False``
    """

    def __init__(
        self,
        neg_slope: float = 0.0,
        mode: str = "fan_in",
        nonlinearity: str = "leaky_relu",
        learnable_only: bool = True,
        log_info: bool = False,
    ) -> None:
        super().__init__()
        self._neg_slope = float(neg_slope)
        self._mode = str(mode)
        self._nonlinearity = str(nonlinearity)
        self._learnable_only = bool(learnable_only)
        self._log_info = bool(log_info)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(mode={self._mode}, nonlinearity={self._nonlinearity}, "
            f"neg_slope={self._neg_slope}, learnable_only={self._learnable_only}, "
            f"log_info={self._log_info})"
        )


class KaimingNormal(BaseKaiming):
    r"""Implements a module parameter initializer with the Kaiming Normal
    strategy.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.nn.init import KaimingNormal
        >>> from torch import nn
        >>> module = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.BatchNorm1d(6), nn.Linear(6, 1))
        >>> initializer = KaimingNormal()
        >>> initializer
        KaimingNormal(mode=fan_in, nonlinearity=leaky_relu, neg_slope=0.0, learnable_only=True, log_info=False)
        >>> initializer.initialize(module)
    """

    def initialize(self, module: Module) -> None:
        logger.info(
            f"Initializing module parameters with the Kaiming Normal strategy "
            f"(mode={self._mode}, nonlinearity={self._nonlinearity}, neg_slope={self._neg_slope}, "
            f"learnable_only={self._learnable_only})..."
        )
        kaiming_normal(
            module=module,
            neg_slope=self._neg_slope,
            mode=self._mode,
            nonlinearity=self._nonlinearity,
            learnable_only=self._learnable_only,
            log_info=self._log_info,
        )


class KaimingUniform(BaseKaiming):
    r"""Implements a module parameter initializer with the Kaiming
    uniform strategy.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.nn.init import KaimingUniform
        >>> from torch import nn
        >>> module = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.BatchNorm1d(6), nn.Linear(6, 1))
        >>> initializer = KaimingUniform()
        >>> initializer
        KaimingUniform(mode=fan_in, nonlinearity=leaky_relu, neg_slope=0.0, learnable_only=True, log_info=False)
        >>> initializer.initialize(module)
    """

    def initialize(self, module: Module) -> None:
        logger.info(
            f"Initializing module parameters with the Kaiming uniform strategy "
            f"(mode={self._mode}, nonlinearity={self._nonlinearity}, neg_slope={self._neg_slope}, "
            f"learnable_only={self._learnable_only})..."
        )
        kaiming_uniform(
            module=module,
            neg_slope=self._neg_slope,
            mode=self._mode,
            nonlinearity=self._nonlinearity,
            learnable_only=self._learnable_only,
            log_info=self._log_info,
        )


def kaiming_normal(
    module: nn.Module,
    neg_slope: float = 0.0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
    learnable_only: bool = True,
    log_info: bool = False,
) -> None:
    r"""Initializes the module parameters with the Kaiming Normal
    strategy.

    Args:
    ----
        module (``torch.nn.Module``): Specifies the module to
            initialize.
        neg_slope (float, optional): Specifies the negative slope of
            the rectifier used after this layer (only used with
            ``'leaky_relu'``). Default: ``0.0``
        mode (str, optional): either ``'fan_in'`` or ``'fan_out'``.
            Choosing ``'fan_in'`` preserves the magnitude of the
            variance of the weights in the forward pass. Choosing
            ``'fan_out'`` preserves the magnitudes in the backwards
            pass. Default: ``'fan_in'``
        nonlinearity (str, optional): the non-linear function
            (`nn.functional` name), recommended to use only with
            ``'relu'`` or ``'leaky_relu'``. Default: ``'leaky_relu'``
        learnable_only (bool, optional): If ``True``, only the
            learnable parameters are initialized, otherwise all the
            parameters are initialized. Default: ``True``
        log_info (bool, optional): If ``True``, log some information
            about the weights that are initialized. Default: ``False``

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.nn.init import kaiming_normal
        >>> from torch import nn
        >>> module = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.BatchNorm1d(6), nn.Linear(6, 1))
        >>> kaiming_normal(module)
    """
    for name, params in module.named_parameters():
        if params.ndim > 1 and (not learnable_only or learnable_only and params.requires_grad):
            if log_info:
                logger.info(
                    f"Initializing '{name}' with Kaiming Normal (mode={mode}, "
                    f"nonlinearity={nonlinearity}, neg_slope={neg_slope}) | shape={params.shape}"
                )
            nn.init.kaiming_normal_(params.data, a=neg_slope, mode=mode, nonlinearity=nonlinearity)


def kaiming_uniform(
    module: nn.Module,
    neg_slope: float = 0.0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
    learnable_only: bool = True,
    log_info: bool = False,
) -> None:
    r"""Initializes the module parameters with the Kaiming uniform
    strategy.

    Args:
    ----
        module (``torch.nn.Module``): Specifies the module to
            initialize.
        neg_slope (float, optional): Specifies the negative slope of
            the rectifier used after this layer (only used with
            ``'leaky_relu'``). Default: ``0.0``
        mode (str, optional): either ``'fan_in'`` or ``'fan_out'``.
            Choosing ``'fan_in'`` preserves the magnitude of the
            variance of the weights in the forward pass. Choosing
            ``'fan_out'`` preserves the magnitudes in the backwards
            pass. Default: ``'fan_in'``
        nonlinearity (str, optional): the non-linear function
            (`nn.functional` name), recommended to use only with
            ``'relu'`` or ``'leaky_relu'``. Default: ``'leaky_relu'``
        learnable_only (bool, optional): If ``True``, only the
            learnable parameters are initialized, otherwise all the
            parameters are initialized. Default: ``True``
        log_info (bool, optional): If ``True``, log some information
            about the weights that are initialized. Default: ``False``

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.nn.init import kaiming_uniform
        >>> from torch import nn
        >>> module = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.BatchNorm1d(6), nn.Linear(6, 1))
        >>> kaiming_uniform(module)
    """
    for name, params in module.named_parameters():
        if params.ndim > 1 and (not learnable_only or learnable_only and params.requires_grad):
            if log_info:
                logger.info(
                    f"Initializing '{name}' with Kaiming uniform (mode={mode}, "
                    f"nonlinearity={nonlinearity}, neg_slope={neg_slope}) | shape={params.shape}"
                )
            nn.init.kaiming_uniform_(params.data, a=neg_slope, mode=mode, nonlinearity=nonlinearity)
