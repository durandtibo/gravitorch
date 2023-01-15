from __future__ import annotations

__all__ = ["BaseModelCreator", "setup_model_creator"]

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from objectory import AbstractFactory
from torch import nn

from gravitorch.utils.format import str_target_object

if TYPE_CHECKING:
    from gravitorch.engines import BaseEngine

logger = logging.getLogger(__name__)


class BaseModelCreator(ABC, metaclass=AbstractFactory):
    r"""Defines the base class to create a model.

    Note that it is not the unique approach to create a model. Feel free
    to use other approaches if this approach does not fit your needs.
    """

    @abstractmethod
    def create(self, engine: BaseEngine) -> nn.Module:
        r"""Creates a model on the device(s) where it should run.

        This method is responsible to register the event handlers
        associated to the model. This method is also responsible to
        move the model parameters to the device(s).

        Args:
            engine (``gravitorch.engines.BaseEngine``): Specifies an
                engine.

        Returns:
            ``torch.nn.Module``: The created model.
        """


def setup_model_creator(creator: BaseModelCreator | dict) -> BaseModelCreator:
    r"""Sets up the model creator.

    The model creator is instantiated from its configuration by using
    the ``BaseModelCreator`` factory function.

    Args:
        creator (``BaseModelCreator`` or dict): Specifies the model
            creator or its configuration.

    Returns:
        ``BaseModelCreator``: The instantiated model creator.
    """
    if isinstance(creator, dict):
        logger.info(
            f"Initializing a model creator from its configuration... {str_target_object(creator)}"
        )
        creator = BaseModelCreator.factory(**creator)
    return creator
