__all__ = ["VanillaDataSourceCreator"]

import logging

from gravitorch import constants as ct
from gravitorch.creators.datasource.base import BaseDataSourceCreator
from gravitorch.datasources.base import BaseDataSource
from gravitorch.datasources.utils import setup_and_attach_data_source, setup_data_source
from gravitorch.engines.base import BaseEngine

logger = logging.getLogger(__name__)


class VanillaDataSourceCreator(BaseDataSourceCreator):
    r"""Implements a simple data source creator.

    Args:
        config (dict): Specifies the data source configuration.
        attach_to_engine (bool, optional): If ``True``, the data
            source is attached to the engine. Default: ``True``
        add_module_to_engine (bool, optional): If ``True``, the data
            source is added to the engine state, so the data source
            state is stored when the engine creates a checkpoint.
            Default: ``True``
    """

    def __init__(
        self,
        config: dict,
        attach_to_engine: bool = True,
        add_module_to_engine: bool = True,
    ):
        self._config = config
        self._attach_to_engine = bool(attach_to_engine)
        self._add_module_to_engine = bool(add_module_to_engine)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(attach_to_engine={self._attach_to_engine}, "
            f"add_module_to_engine={self._add_module_to_engine})"
        )

    def create(self, engine: BaseEngine) -> BaseDataSource:
        logger.info("Creating a data source...")
        if self._attach_to_engine:
            data_source = setup_and_attach_data_source(data_source=self._config, engine=engine)
        else:
            data_source = setup_data_source(data_source=self._config)
        if self._add_module_to_engine:
            logger.info(f"Adding a data source to the engine (key: {ct.DATA_SOURCE})...")
            engine.add_module(ct.DATA_SOURCE, data_source)
        return data_source
