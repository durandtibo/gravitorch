__all__ = ["TruncNormalParameterInitializer", "recursive_trunc_normal_"]

import logging
from typing import Union

from torch import nn

from gravitorch.engines.base import BaseEngine
from gravitorch.utils.parameter_initializers import BaseDefaultParameterInitializer

logger = logging.getLogger(__name__)


class TruncNormalParameterInitializer(BaseDefaultParameterInitializer):
    r"""Implements a model parameter initializer with a truncated Normal
    strategy.

    Args:
        mean (int or float, optional): Specifies the mean of the
            Normal distribution. Default: ``0.0``
        std (int or float, optional): Specifies the standard
            deviation of the Normal distribution. Default: ``1.0``
        min_cutoff (int or float, optional): Specifies the minimum
            cutoff value. Default: ``-2.0``
        max_cutoff (int or float, optional): Specifies the maximum
            cutoff value. Default: ``2.0``
        learnable_only (bool, optional): If ``True``, only the
            learnable parameters are initialized, otherwise all the
            parameters are initialized. Default: ``True``
        show_stats (bool, optional): If ``True``, the parameter
            statistics are shown at the end of the initialization.
            Default: ``True``
    """

    def __init__(
        self,
        mean: Union[int, float] = 0.0,
        std: Union[int, float] = 1.0,
        min_cutoff: Union[int, float] = -2.0,
        max_cutoff: Union[int, float] = 2.0,
        learnable_only: bool = True,
        show_stats: bool = True,
    ):
        super().__init__(show_stats=show_stats)
        self._mean = float(mean)
        self._std = float(std)
        self._min_cutoff = float(min_cutoff)
        self._max_cutoff = float(max_cutoff)
        self._learnable_only = bool(learnable_only)

    def __str__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(mean={self._mean}, std={self._std}, "
            f"min_cutoff={self._min_cutoff}, max_cutoff={self._max_cutoff}, "
            f"learnable_only={self._learnable_only})"
        )

    def _initialize(self, engine: BaseEngine) -> None:
        logger.info(
            f"Initialize model parameters with a truncated Normal strategy (mean={self._mean:,.6f}, "
            f"std={self._std:,.6f}, min_cutoff={self._min_cutoff:,.6f}, max_cutoff={self._max_cutoff:,.6f})"
        )
        recursive_trunc_normal_(
            module=engine.model,
            mean=self._mean,
            std=self._std,
            min_cutoff=self._min_cutoff,
            max_cutoff=self._max_cutoff,
            learnable_only=self._learnable_only,
        )


def recursive_trunc_normal_(
    module: nn.Module,
    mean: float = 0.0,
    std: float = 1.0,
    min_cutoff: float = -2.0,
    max_cutoff: float = 2.0,
    learnable_only: bool = True,
) -> None:
    r"""Initialize the parameters of the module with the truncated Normal
    initialization.

    Args:
        module (``torch.nn.Module``): Specifies the module to
            initialize.
        mean (float, optional): Specifies the mean of the Normal
            distribution. Default: ``0.0``
        std (float, optional): Specifies the standard deviation of the
            Normal distribution. Default: ``1.0``
        min_cutoff (float, optional): Specifies the minimum cutoff
            value. Default: ``-2.0``
        max_cutoff (float, optional): Specifies the maximum cutoff
            value. Default: ``2.0``
        learnable_only (bool, optional): If ``True``, only the
            learnable parameters are initialized, otherwise all the
            parameters are initialized. Default: ``True``

    Example usage:

    .. code-block:: python

        >>> from gravitorch.utils.parameter_initializers import recursive_trunc_normal_
        >>> from torch import nn
        >>> net = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.BatchNorm1d(6), nn.Linear(6, 1))
        >>> recursive_trunc_normal_(net)
    """
    for params in module.parameters():
        if not learnable_only or learnable_only and params.requires_grad:
            nn.init.trunc_normal_(params.data, mean=mean, std=std, a=min_cutoff, b=max_cutoff)
