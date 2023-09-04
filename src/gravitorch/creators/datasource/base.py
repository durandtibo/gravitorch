from __future__ import annotations

__all__ = ["BaseDataSourceCreator", "setup_datasource_creator"]

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from objectory import AbstractFactory

from gravitorch.utils.format import str_target_object

if TYPE_CHECKING:
    from gravitorch.datasources import (
        BaseDataSource,  # TODO: incorrect because used in setup
    )
    from gravitorch.engines import BaseEngine

logger = logging.getLogger(__name__)


class BaseDataSourceCreator(ABC, metaclass=AbstractFactory):
    r"""Defines the base class to create a datasource.

    Note that it is not the unique approach to create a datasource. Feel
    free to use other approaches if this approach does not fit your
    needs.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.testing import create_dummy_engine
        >>> from gravitorch.creators.datasource import VanillaDataSourceCreator
        >>> creator = VanillaDataSourceCreator({"_target_": "gravitorch.testing.DummyDataSource"})
        >>> creator
        VanillaDataSourceCreator(attach_to_engine=True, add_module_to_engine=True)
        >>> engine = create_dummy_engine()
        >>> datasource = creator.create(engine)
        >>> datasource
        DummyDataSource(
          datasets:
            (train): DummyDataset(num_examples=4, feature_size=4)
            (eval): DummyDataset(num_examples=4, feature_size=4)
          dataloader_creators:
            (train): DataLoaderCreator(
                batch_size : 1
                seed       : 0
                shuffle    : False
              )
            (eval): DataLoaderCreator(
                batch_size : 1
                seed       : 0
                shuffle    : False
              )
        )
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

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.testing import create_dummy_engine
            >>> from gravitorch.creators.datasource import VanillaDataSourceCreator
            >>> creator = VanillaDataSourceCreator({"_target_": "gravitorch.testing.DummyDataSource"})
            >>> engine = create_dummy_engine()
            >>> datasource = creator.create(engine)
            >>> datasource
            DummyDataSource(
              datasets:
                (train): DummyDataset(num_examples=4, feature_size=4)
                (eval): DummyDataset(num_examples=4, feature_size=4)
              dataloader_creators:
                (train): DataLoaderCreator(
                    batch_size : 1
                    seed       : 0
                    shuffle    : False
                  )
                (eval): DataLoaderCreator(
                    batch_size : 1
                    seed       : 0
                    shuffle    : False
                  )
            )
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

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.creators.datasource import setup_datasource_creator
        >>> creator = setup_datasource_creator(
        ...     {
        ...         "_target_": "gravitorch.creators.datasource.VanillaDataSourceCreator",
        ...         "config": {"_target_": "gravitorch.testing.DummyDataSource"},
        ...     }
        ... )
        >>> creator
        VanillaDataSourceCreator(attach_to_engine=True, add_module_to_engine=True)
    """
    if isinstance(creator, dict):
        logger.info(
            "Initializing the datasource creator from its configuration... "
            f"{str_target_object(creator)}"
        )
        creator = BaseDataSourceCreator.factory(**creator)
    return creator
