__all__ = [
    "ConstantBiasParameterInitializer",
    "bias_constant_",
    "recursive_bias_constant_",
    "recursive_constant_",
]

import logging
from typing import Union

from torch import nn

from gravitorch.engines.base import BaseEngine
from gravitorch.utils.parameter_initializers.base import BaseDefaultParameterInitializer

logger = logging.getLogger(__name__)


class ConstantBiasParameterInitializer(BaseDefaultParameterInitializer):
    r"""Implements a model parameter initializer with the Xavier Normal or
    uniform strategy.

    Args:
        value (float): Specifies the value to initialize the
            parameters with.
        learnable_only (bool, optional): If ``True``, only the
            learnable parameters are initialized, otherwise all the
            parameters are initialized. Default: ``True``
        log_info (bool, optional): If ``True``, log some information
            about the weights that are initialized. Default: ``False``
        show_stats (bool, optional): If ``True``, the parameter
            statistics are shown at the end of the initialization.
            Default: ``True``
    """

    def __init__(
        self,
        value: Union[int, float] = 0,
        learnable_only: bool = True,
        log_info: bool = False,
        show_stats: bool = True,
    ):
        super().__init__(show_stats=show_stats)
        self._value = float(value)
        self._learnable_only = bool(learnable_only)
        self._log_info = bool(log_info)

    def __str__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(value={self._value}, "
            f"learnable_only={self._learnable_only}, log_info={self._log_info})"
        )

    def _initialize(self, engine: BaseEngine) -> None:
        if self._learnable_only:
            logger.info(f"Initializing the learnable biases to {self._value}...")
        else:
            logger.info(f"Initializing all biases to {self._value}...")
        recursive_bias_constant_(
            module=engine.model,
            value=self._value,
            learnable_only=self._learnable_only,
            log_info=self._log_info,
        )


def bias_constant_(module: nn.Module, value: float) -> None:
    r"""Initializes the biases of some modules to ``value``.

    This function initializes the biases of the following modules:

        - ``torch.nn.Linear``
        - ``torch.nn.Bilinear``
        - ``torch.nn.ConvNd``
        - ``torch.nn.LayerNorm``
        - ``torch.nn.GroupNorm``
        - ``torch.nn.BatchNormNd``

    Args:
        module (``torch.nn.Module``): Specifies the module to
            initialize.
        value (float): Specifies the value to initialize the
            parameters with.

    Example usage:

    .. code-block:: python

        >>> from gravitorch.utils.parameter_initializers import bias_constant_
        >>> from torch import nn
        >>> layer = nn.Linear(4, 6)
        >>> bias_constant_(layer, value=2)
        >>> layer.weight
        Parameter containing:
        tensor([[ 0.1179, -0.2250, -0.0784, -0.0855],
                [-0.3958, -0.4466, -0.3180, -0.0501],
                [-0.3657, -0.3836, -0.2601,  0.3431],
                [-0.2073, -0.2916,  0.1014,  0.4463],
                [ 0.1758,  0.1229,  0.2563,  0.2391],
                [ 0.2957,  0.1145,  0.3906, -0.2804]], requires_grad=True)
        >>> layer.bias
        Parameter containing:
        tensor([2., 2., 2., 2., 2., 2.], requires_grad=True)
    """
    layers_with_bias = (
        nn.Linear,
        nn.Bilinear,
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
        nn.LayerNorm,
        nn.GroupNorm,
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
    )
    if isinstance(module, layers_with_bias) and module.bias is not None:
        nn.init.constant_(module.bias, value)
    elif isinstance(module, nn.MultiheadAttention):
        for bias in [module.in_proj_bias, module.bias_k, module.bias_v, module.out_proj.bias]:
            if bias is not None:
                nn.init.constant_(bias, value)


def recursive_bias_constant_(
    module: nn.Module,
    value: Union[int, float],
    learnable_only: bool = True,
    log_info: bool = False,
) -> None:
    r"""Recursively initialize the biases with ``value``.

    To identify the biases, this function looks at if ``'bias'`` is
    in the parameter name.

    Args:
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

        >>> from gravitorch.utils.parameter_initializers import recursive_bias_constant_
        >>> from torch import nn
        >>> net = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.BatchNorm1d(6), nn.Linear(6, 1))
        >>> recursive_bias_constant_(net, 2)
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
            nn.init.constant_(params.data, value)
            if log_info:
                logger.info(f"Initialize '{name}' to {value} | shape={params.shape}")


def recursive_constant_(
    module: nn.Module, value: Union[int, float], learnable_only: bool = True
) -> None:
    r"""Initialize the parameters of the module with ``value``.

    Args:
        module (``torch.nn.Module``): Specifies the module with the
            parameters to initialize.
        value (int, float): Specifies the value to initialize the
            parameters with.
        learnable_only (bool, optional): If ``True``, only the
            learnable parameters are initialized, otherwise all the
            parameters are initialized. Default: ``True``

    Example usage:

    .. code-block:: python

        >>> from gravitorch.utils.parameter_initializers import recursive_constant_
        >>> from torch import nn
        >>> net = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.BatchNorm1d(6), nn.Linear(6, 1))
        >>> recursive_constant_(net, 2)
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
    for params in module.parameters():
        if not learnable_only or learnable_only and params.requires_grad:
            nn.init.constant_(params.data, value)
