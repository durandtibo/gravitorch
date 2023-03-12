__all__ = ["NoOpInitializer"]

import logging

from torch.nn import Module

from gravitorch.nn.init.base import BaseInitializer

logger = logging.getLogger(__name__)


class NoOpInitializer(BaseInitializer):
    r"""This is the special class that does not update the module parameters.

    You should use this class if the parameters of the module are
    initialized somewhere else.
    """

    def __str__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def initialize(self, module: Module) -> None:
        logger.info("The module parameters are not updated")
