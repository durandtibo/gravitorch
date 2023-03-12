__all__ = [
    "Constant",
    "ConstantBias",
    "constant_init",
    "constant_bias_init",
]

import logging
from typing import Union

from torch import nn
from torch.nn import Module

from gravitorch.nn.init.base import BaseInitializer

logger = logging.getLogger(__name__)


class Constant(BaseInitializer):
    r"""Implements a module parameter initializer where the weights are
    initialized with constant values.

    Args:
    ----
        value (float): Specifies the value to initialize the
            parameters with.
        learnable_only (bool, optional): If ``True``, only the
            learnable parameters are initialized, otherwise all the
            parameters are initialized. Default: ``True``
        log_info (bool, optional): If ``True``, log some information
            about the weights that are initialized. Default: ``False``
    """

    def __init__(
        self,
        value: Union[int, float] = 0.0,
        learnable_only: bool = True,
        log_info: bool = False,
    ) -> None:
        super().__init__()
        self._value = float(value)
        self._learnable_only = bool(learnable_only)
        self._log_info = bool(log_info)

    def __str__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(value={self._value}, "
            f"learnable_only={self._learnable_only}, log_info={self._log_info})"
        )

    def initialize(self, module: Module) -> None:
        logger.info(
            f"Initializing weights with {self._value} (learnable_only: {self._learnable_only})..."
        )
        constant_init(
            module=module,
            value=self._value,
            learnable_only=self._learnable_only,
            log_info=self._log_info,
        )


class ConstantBias(BaseInitializer):
    r"""Implements a module parameter initializer where the biases are
    initialized with constant values.

    Args:
    ----
        value (float): Specifies the value to initialize the
            parameters with.
        learnable_only (bool, optional): If ``True``, only the
            learnable parameters are initialized, otherwise all the
            parameters are initialized. Default: ``True``
        log_info (bool, optional): If ``True``, log some information
            about the biases that are initialized. Default: ``False``
    """

    def __init__(
        self,
        value: Union[int, float] = 0.0,
        learnable_only: bool = True,
        log_info: bool = False,
    ) -> None:
        super().__init__()
        self._value = float(value)
        self._learnable_only = bool(learnable_only)
        self._log_info = bool(log_info)

    def __str__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(value={self._value}, "
            f"learnable_only={self._learnable_only}, log_info={self._log_info})"
        )

    def initialize(self, module: Module) -> None:
        logger.info(
            f"Initializing biases with {self._value} (learnable_only: {self._learnable_only})..."
        )
        constant_bias_init(
            module=module,
            value=self._value,
            learnable_only=self._learnable_only,
            log_info=self._log_info,
        )


def constant_bias_init(
    module: Module,
    value: Union[int, float],
    learnable_only: bool = True,
    log_info: bool = False,
) -> None:
    r"""Recursively initialize the biases with ``value``.

    To identify the biases, this function looks at if ``'bias'`` is
    in the parameter name.

    Args:
    ----
        module (``torch.nn.Module``): Specifies the module to
            initialize.
        value (float): Specifies the value to initialize the
            parameters with.
        learnable_only (bool, optional): If ``True``, only the
            learnable parameters are initialized, otherwise all the
            parameters are initialized. Default: ``True``
        log_info (bool, optional): If ``True``, log some information
            about the weights that are initialized. Default: ``False``

    Example usage:

    .. code-block:: python

        >>> from gravitorch.nn.init import constant_bias_init
        >>> from torch import nn
        >>> net = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.BatchNorm1d(6), nn.Linear(6, 1))
        >>> constant_bias_init(net, 2)
        >>> for key, param in net.named_parameters():
        ...     print(key, param)
        0.weight Parameter containing:
        tensor([[ 0.1911,  0.2128, -0.3738,  0.3777],
                [ 0.3394,  0.1795, -0.1418,  0.1076],
                [-0.2050, -0.0029, -0.2555,  0.2612],
                [ 0.1260, -0.3503,  0.1254, -0.3669],
                [ 0.1261, -0.1652,  0.1695,  0.3879],
                [-0.0405,  0.0372,  0.0515, -0.4335]], requires_grad=True)
        0.bias Parameter containing:
        tensor([2., 2., 2., 2., 2., 2.], requires_grad=True)
        2.weight Parameter containing:
        tensor([1., 1., 1., 1., 1., 1.], requires_grad=True)
        2.bias Parameter containing:
        tensor([2., 2., 2., 2., 2., 2.], requires_grad=True)
        3.weight Parameter containing:
        tensor([[-0.2702,  0.2759, -0.2408,  0.1202, -0.1770,  0.0223]],
               requires_grad=True)
        3.bias Parameter containing:
        tensor([2.], requires_grad=True)
    """
    for name, params in module.named_parameters():
        if "bias" in name and (not learnable_only or learnable_only and params.requires_grad):
            if log_info:
                logger.info(f"Initializing bias '{name}' with {value} | shape={params.shape}")
            nn.init.constant_(params.data, value)


def constant_init(
    module: Module,
    value: Union[int, float],
    learnable_only: bool = True,
    log_info: bool = False,
) -> None:
    r"""Initializes the parameters of the module with a constant ``value``.

    Args:
    ----
        module (``torch.nn.Module``): Specifies the module with the
            parameters to initialize.
        value (int, float): Specifies the value to initialize the
            parameters with.
        learnable_only (bool, optional): If ``True``, only the
            learnable parameters are initialized, otherwise all the
            parameters are initialized. Default: ``True``
        log_info (bool, optional): If ``True``, log some information
            about the weights that are initialized. Default: ``False``

    Example usage:

    .. code-block:: python

        >>> from gravitorch.nn.init import constant_init
        >>> from torch import nn
        >>> net = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.BatchNorm1d(6), nn.Linear(6, 1))
        >>> constant_init(net, 2.0)
        >>> for key, param in net.named_parameters():
        ...     print(key, param)
        0.weight Parameter containing:
        tensor([[2., 2., 2., 2.],
                [2., 2., 2., 2.],
                [2., 2., 2., 2.],
                [2., 2., 2., 2.],
                [2., 2., 2., 2.],
                [2., 2., 2., 2.]], requires_grad=True)
        0.bias Parameter containing:
        tensor([2., 2., 2., 2., 2., 2.], requires_grad=True)
        2.weight Parameter containing:
        tensor([2., 2., 2., 2., 2., 2.], requires_grad=True)
        2.bias Parameter containing:
        tensor([2., 2., 2., 2., 2., 2.], requires_grad=True)
        3.weight Parameter containing:
        tensor([[2., 2., 2., 2., 2., 2.]], requires_grad=True)
        3.bias Parameter containing:
        tensor([2.], requires_grad=True)
    """
    for name, params in module.named_parameters():
        if not learnable_only or learnable_only and params.requires_grad:
            if log_info:
                logger.info(f"Initializing '{name}' with {value} | shape={params.shape}")
            nn.init.constant_(params.data, value)
