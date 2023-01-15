from __future__ import annotations

__all__ = ["BaseDataSourceCreator", "setup_data_source_creator"]

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from objectory import AbstractFactory

from gravitorch.utils.format import str_target_object

if TYPE_CHECKING:
    from gravitorch.datasources import BaseDataSource
    from gravitorch.engines import BaseEngine

logger = logging.getLogger(__name__)


class BaseDataSourceCreator(ABC, metaclass=AbstractFactory):
    r"""Defines the base class to create a data source.

    Note that it is not the unique approach to create a data source.
    Feel free to use other approaches if this approach does not fit your
    needs.
    """

    @abstractmethod
    def create(self, engine: BaseEngine) -> BaseDataSource:
        r"""Creates a data source object.

        This method is responsible to register the event handlers
        associated to the data source.

        Args:
            engine (``gravitorch.engines.BaseEngine``): Specifies an
                engine.

        Returns:
            ``gravitorch.datasources.BaseDataSource``: The created data
                source.
        """


def setup_data_source_creator(creator: BaseDataSourceCreator | dict) -> BaseDataSourceCreator:
    r"""Sets up the data source creator.

    The data source creator is instantiated from its configuration by
    using the ``BaseDataSourceCreator`` factory function.

    Args:
        creator (``BaseDataSourceCreator`` or dict): Specifies the
            data source creator or its configuration.

    Returns:
        ``BaseDataSourceCreator``: The instantiated data source
            creator.
    """
    if isinstance(creator, dict):
        logger.info(
            "Initializing the data source creator from its configuration... "
            f"{str_target_object(creator)}"
        )
        creator = BaseDataSourceCreator.factory(**creator)
    return creator
