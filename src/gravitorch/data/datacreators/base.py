__all__ = ["BaseDataCreator", "setup_data_creator"]

import logging
from abc import ABC, abstractmethod
from typing import Generic, Optional, TypeVar, Union

from objectory import AbstractFactory

from gravitorch.engines.base import BaseEngine
from gravitorch.utils.format import str_target_object

logger = logging.getLogger(__name__)

T = TypeVar("T")


class BaseDataCreator(ABC, Generic[T], metaclass=AbstractFactory):
    r"""Defines the base class to implement a data creator."""

    @abstractmethod
    def create(self, engine: Optional[BaseEngine] = None) -> T:
        r"""Creates data.

        Args:
            engine (``BaseEngine`` or ``None``): Specifies an engine.
                Default: ``None``

        Return:
            The created data.
        """


def setup_data_creator(data_creator: Union[BaseDataCreator, dict]) -> BaseDataCreator:
    r"""Sets up a data creator.

    The data creator is instantiated from its configuration by using
    the ``BaseDataCreator`` factory function.

    Args:
        data_creator (``BaseDataCreator`` or dict): Specifies the data
            creator or its configuration.

    Returns:
        ``BaseDataCreator``: The instantiated data creator.
    """
    if isinstance(data_creator, dict):
        logger.info(
            "Initializing a data creator from its configuration... "
            f"{str_target_object(data_creator)}"
        )
        data_creator = BaseDataCreator.factory(**data_creator)
    return data_creator
