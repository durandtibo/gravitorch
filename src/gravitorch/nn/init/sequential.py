from __future__ import annotations

__all__ = ["SequentialInitializer"]

import logging
from collections.abc import Sequence

from coola.utils import str_indent, str_sequence
from torch.nn import Module

from gravitorch.nn.init.base import BaseInitializer
from gravitorch.nn.init.factory import setup_initializer

logger = logging.getLogger(__name__)


class SequentialInitializer(BaseInitializer):
    r"""Implements a module initializer that sequentially calls module
    initializers.

    Args:
    ----
        initializers: Specifies the sequence of module initializers.
            The sequence order defines the order of the call.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.nn.init import ConstantBias, KaimingNormal, SequentialInitializer
        >>> from torch import nn
        >>> module = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.BatchNorm1d(6), nn.Linear(6, 1))
        >>> initializer = SequentialInitializer([KaimingNormal(), ConstantBias(value=0.0)])
        >>> initializer
        SequentialInitializer(
          (0): KaimingNormal(mode=fan_in, nonlinearity=leaky_relu, neg_slope=0.0, learnable_only=True, log_info=False)
          (1): ConstantBias(value=0.0, learnable_only=True, log_info=False)
        )
        >>> initializer.initialize(module)
    """

    def __init__(
        self,
        initializers: Sequence[BaseInitializer | dict],
    ) -> None:
        super().__init__()
        self._initializers = tuple(setup_initializer(initializer) for initializer in initializers)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  {str_indent(str_sequence(self._initializers))}\n)"
        )

    def initialize(self, module: Module) -> None:
        for i, initializer in enumerate(self._initializers):
            logger.info(f"[{i}/{len(self._initializers)}] {initializer}")
            initializer.initialize(module)
