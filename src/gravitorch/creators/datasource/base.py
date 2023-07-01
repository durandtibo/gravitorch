from __future__ import annotations

__all__ = ["BaseDataSourceCreator", "setup_datasource_creator"]

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
    r"""Defines the base class to create a datasource.

    Note that it is not the unique approach to create a datasource. Feel
    free to use other approaches if this approach does not fit your
    needs.
    """

    @abstractmethod
    def create(self, engine: BaseEngine) -> BaseDataSource:
        r"""Creates a datasource object.

        This method is responsible to register the event handlers
        associated to the datasource.

        Args:
        ----
            engine (``gravitorch.engines.BaseEngine``): Specifies an
                engine.

        Returns:
        -------
            ``gravitorch.datasources.BaseDataSource``: The created data
                source.
        """


def setup_datasource_creator(creator: BaseDataSourceCreator | dict) -> BaseDataSourceCreator:
    r"""Sets up the datasource creator.

    The datasource creator is instantiated from its configuration by
    using the ``BaseDataSourceCreator`` factory function.

    Args:
    ----
        creator (``BaseDataSourceCreator`` or dict): Specifies the
            datasource creator or its configuration.

    Returns:
    -------
        ``BaseDataSourceCreator``: The instantiated datasource
            creator.
    """
    if isinstance(creator, dict):
        logger.info(
            "Initializing the datasource creator from its configuration... "
            f"{str_target_object(creator)}"
        )
        creator = BaseDataSourceCreator.factory(**creator)
    return creator
