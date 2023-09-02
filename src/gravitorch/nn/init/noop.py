from __future__ import annotations

__all__ = ["NoOpInitializer"]

import logging

from torch.nn import Module

from gravitorch.nn.init.base import BaseInitializer

logger = logging.getLogger(__name__)


class NoOpInitializer(BaseInitializer):
    r"""This is the special class that does not update the module
    parameters.

    You should use this class if the parameters of the module are
    initialized somewhere else.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.nn.init import NoOpInitializer
        >>> from torch import nn
        >>> module = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.BatchNorm1d(6), nn.Linear(6, 1))
        >>> initializer = NoOpInitializer()
        >>> initializer
        NoOpInitializer()
        >>> initializer.initialize(module)
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def initialize(self, module: Module) -> None:
        logger.info("The module parameters are not updated")
