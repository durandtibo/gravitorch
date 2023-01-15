__all__ = ["NoParameterInitializer"]

import logging

from gravitorch.engines.base import BaseEngine
from gravitorch.utils.parameter_initializers.base import BaseDefaultParameterInitializer

logger = logging.getLogger(__name__)


class NoParameterInitializer(BaseDefaultParameterInitializer):
    r"""This is the special class that does not update the model parameters.

    You should use this class if the parameters of the models are
    initialized somewhere else.
    """

    def __str__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def _initialize(self, engine: BaseEngine) -> None:
        logger.info("The model parameters are not updated")
