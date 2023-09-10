from __future__ import annotations

__all__ = ["NoOptimizerCreator"]

import logging
from typing import TYPE_CHECKING

from torch.nn import Module

from gravitorch.creators.optimizer.base import BaseOptimizerCreator

if TYPE_CHECKING:
    from gravitorch.engines import BaseEngine

logger = logging.getLogger(__name__)


class NoOptimizerCreator(BaseOptimizerCreator):
    r"""Implements a no optimizer creator.

    This optimizer creator should be used if you do not want to create
    an optimizer. For example if you only want to evaluate your model,
    you do not need to create an optimizer. The ``create`` method always
    returns ``None``.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.testing import create_dummy_engine, DummyClassificationModel
        >>> from gravitorch.creators.optimizer import OptimizerCreator
        >>> creator = NoOptimizerCreator()
        >>> creator
        NoOptimizerCreator()
        >>> engine = create_dummy_engine()
        >>> model = DummyClassificationModel()
        >>> optimizer = creator.create(engine, model)
        >>> optimizer
        None
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def create(self, engine: BaseEngine, model: Module) -> None:
        logger.info("No optimizer")
        return
