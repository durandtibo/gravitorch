from __future__ import annotations

__all__ = ["TruncNormal", "trunc_normal"]

import logging

from torch import nn
from torch.nn import Module

from gravitorch.nn.init.base import BaseInitializer

logger = logging.getLogger(__name__)


class TruncNormal(BaseInitializer):
    r"""Implements a module parameter initializer with a truncated Normal
    strategy.

    Args:
    ----
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

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.nn.init import TruncNormal
        >>> from torch import nn
        >>> module = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.BatchNorm1d(6), nn.Linear(6, 1))
        >>> initializer = TruncNormal()
        >>> initializer
        TruncNormal(mean=0.0, std=1.0, min_cutoff=-2.0, max_cutoff=2.0, learnable_only=True, log_info=False)
        >>> initializer.initialize(module)
    """

    def __init__(
        self,
        mean: int | float = 0.0,
        std: int | float = 1.0,
        min_cutoff: int | float = -2.0,
        max_cutoff: int | float = 2.0,
        learnable_only: bool = True,
        log_info: bool = False,
    ) -> None:
        super().__init__()
        self._mean = float(mean)
        self._std = float(std)
        self._min_cutoff = float(min_cutoff)
        self._max_cutoff = float(max_cutoff)
        self._learnable_only = bool(learnable_only)
        self._log_info = bool(log_info)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(mean={self._mean}, std={self._std}, "
            f"min_cutoff={self._min_cutoff}, max_cutoff={self._max_cutoff}, "
            f"learnable_only={self._learnable_only}, log_info={self._log_info})"
        )

    def initialize(self, module: Module) -> None:
        logger.info(
            f"Initialize module parameters with a truncated Normal strategy "
            f"(mean={self._mean:,.6f}, std={self._std:,.6f}, "
            f"min_cutoff={self._min_cutoff:,.6f}, max_cutoff={self._max_cutoff:,.6f},"
            f"learnable_only={self._learnable_only})"
        )
        trunc_normal(
            module=module,
            mean=self._mean,
            std=self._std,
            min_cutoff=self._min_cutoff,
            max_cutoff=self._max_cutoff,
            learnable_only=self._learnable_only,
            log_info=self._log_info,
        )


def trunc_normal(
    module: Module,
    mean: float = 0.0,
    std: float = 1.0,
    min_cutoff: float = -2.0,
    max_cutoff: float = 2.0,
    learnable_only: bool = True,
    log_info: bool = False,
) -> None:
    r"""Initializes the module parameters with the truncated Normal
    strategy.

    Args:
    ----
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
        log_info (bool, optional): If ``True``, log some information
            about the weights that are initialized. Default: ``False``

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.nn.init import trunc_normal
        >>> from torch import nn
        >>> module = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.BatchNorm1d(6), nn.Linear(6, 1))
        >>> trunc_normal(module)
    """
    for name, params in module.named_parameters():
        if not learnable_only or learnable_only and params.requires_grad:
            if log_info:
                logger.info(
                    f"Initializing '{name}' with truncated Normal (mean={mean:.6f}, "
                    f"std={std:.6f}, min_cutoff={min_cutoff:.6f}, max_cutoff={max_cutoff:.6f}) "
                    f"| shape={params.shape}"
                )
            nn.init.trunc_normal_(params.data, mean=mean, std=std, a=min_cutoff, b=max_cutoff)
