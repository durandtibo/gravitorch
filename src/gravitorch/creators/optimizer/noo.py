__all__ = ["NoOptimizerCreator"]

import logging

from torch.nn import Module

from gravitorch.creators.optimizer.base import BaseOptimizerCreator
from gravitorch.engines.base import BaseEngine

logger = logging.getLogger(__name__)


class NoOptimizerCreator(BaseOptimizerCreator):
    r"""Implements a no optimizer creator.

    This optimizer creator should be used if you do not want to create
    an optimizer. For example if you only want to evaluate your model,
    you do not need to create an optimizer. The ``create`` method always
    returns ``None``.
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def create(self, engine: BaseEngine, model: Module) -> None:
        r"""Does not create an optimizer.

        Args:
            engine (``gravitorch.engines.BaseEngine``): Specifies an
                engine.
            model (``torch.nn.Module``): Specifies a model.

        Returns:
            ``None``: because there is no optimizer to create.
        """
        logger.info("No optimizer")
        return None
