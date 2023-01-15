__all__ = [
    "BaseKaimingParameterInitializer",
    "KaimingNormalParameterInitializer",
    "KaimingUniformParameterInitializer",
    "recursive_kaiming_normal_",
    "recursive_kaiming_uniform_",
]

import logging

from torch import nn

from gravitorch.engines.base import BaseEngine
from gravitorch.utils.parameter_initializers.base import BaseDefaultParameterInitializer

logger = logging.getLogger(__name__)


class BaseKaimingParameterInitializer(BaseDefaultParameterInitializer):
    r"""Implements a model parameter initializer with the Kaiming Normal or
    uniform strategy.

    Args:
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
        show_stats (bool, optional): If ``True``, the parameter
            statistics are shown at the end of the initialization.
            Default: ``True``
    """

    def __init__(
        self,
        neg_slope: float = 0.0,
        mode: str = "fan_in",
        nonlinearity: str = "leaky_relu",
        learnable_only: bool = True,
        show_stats: bool = True,
    ):
        super().__init__(show_stats=show_stats)
        self._neg_slope = float(neg_slope)
        self._mode = str(mode)
        self._nonlinearity = str(nonlinearity)
        self._learnable_only = bool(learnable_only)

    def __str__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(neg_slope={self._neg_slope}, mode={self._mode}, "
            f"nonlinearity={self._nonlinearity}, learnable_only={self._learnable_only})"
        )


class KaimingNormalParameterInitializer(BaseKaimingParameterInitializer):
    r"""Implements a model parameter initializer with the Kaiming Normal
    strategy."""

    def _initialize(self, engine: BaseEngine) -> None:
        logger.info("Initializing model parameters with the Kaiming Normal strategy...")
        recursive_kaiming_normal_(
            module=engine.model,
            neg_slope=self._neg_slope,
            mode=self._mode,
            nonlinearity=self._nonlinearity,
            learnable_only=self._learnable_only,
        )


class KaimingUniformParameterInitializer(BaseKaimingParameterInitializer):
    r"""Implements a model parameter initializer with the Kaiming uniform
    strategy."""

    def _initialize(self, engine: BaseEngine) -> None:
        logger.info("Initializing model parameters with the Kaiming uniform strategy...")
        recursive_kaiming_uniform_(
            module=engine.model,
            neg_slope=self._neg_slope,
            mode=self._mode,
            nonlinearity=self._nonlinearity,
            learnable_only=self._learnable_only,
        )


def recursive_kaiming_normal_(
    module: nn.Module,
    neg_slope: float = 0.0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
    learnable_only: bool = True,
) -> None:
    r"""Initialize the parameters of the module with the Kaiming Normal
    initialization.

    Args:
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

    Example usage:

    .. code-block:: python

        >>> from gravitorch.utils.parameter_initializers import recursive_kaiming_normal_
        >>> from torch import nn
        >>> net = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.BatchNorm1d(6), nn.Linear(6, 1))
        >>> recursive_kaiming_normal_(net)
    """
    for params in module.parameters():
        if params.ndim > 1 and (not learnable_only or learnable_only and params.requires_grad):
            nn.init.kaiming_normal_(params.data, a=neg_slope, mode=mode, nonlinearity=nonlinearity)


def recursive_kaiming_uniform_(
    module: nn.Module,
    neg_slope: float = 0.0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
    learnable_only: bool = True,
) -> None:
    r"""Initialize the parameters of the module with the Kaiming uniform
    initialization.

    Args:
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
            learnable parameters are initialized, otherwise all
            the parameters are initialized. Default: ``True``

    Example usage:

    .. code-block:: python

        >>> from gravitorch.utils.parameter_initializers import recursive_kaiming_uniform_
        >>> from torch import nn
        >>> net = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.BatchNorm1d(6), nn.Linear(6, 1))
        >>> recursive_kaiming_uniform_(net)
    """
    for params in module.parameters():
        if params.ndim > 1 and (not learnable_only or learnable_only and params.requires_grad):
            nn.init.kaiming_uniform_(params.data, a=neg_slope, mode=mode, nonlinearity=nonlinearity)
