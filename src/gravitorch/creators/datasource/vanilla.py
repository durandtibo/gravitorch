from __future__ import annotations

__all__ = ["VanillaDataSourceCreator"]

import logging
from typing import TYPE_CHECKING

from gravitorch import constants as ct
from gravitorch.creators.datasource.base import BaseDataSourceCreator
from gravitorch.datasources.base import (
    BaseDataSource,
    setup_and_attach_datasource,
    setup_datasource,
)

if TYPE_CHECKING:
    from gravitorch.engines import BaseEngine

logger = logging.getLogger(__name__)


class VanillaDataSourceCreator(BaseDataSourceCreator):
    r"""Implements a simple datasource creator.

    Args:
    ----
        config (dict): Specifies the datasource configuration.
        attach_to_engine (bool, optional): If ``True``, the data
            source is attached to the engine. Default: ``True``
        add_module_to_engine (bool, optional): If ``True``, the data
            source is added to the engine state, so the datasource
            state is stored when the engine creates a checkpoint.
            Default: ``True``
    """

    def __init__(
        self,
        config: dict,
        attach_to_engine: bool = True,
        add_module_to_engine: bool = True,
    ) -> None:
        self._config = config
        self._attach_to_engine = bool(attach_to_engine)
        self._add_module_to_engine = bool(add_module_to_engine)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(attach_to_engine={self._attach_to_engine}, "
            f"add_module_to_engine={self._add_module_to_engine})"
        )

    def create(self, engine: BaseEngine) -> BaseDataSource:
        logger.info("Creating a datasource...")
        if self._attach_to_engine:
            datasource = setup_and_attach_datasource(datasource=self._config, engine=engine)
        else:
            datasource = setup_datasource(datasource=self._config)
        if self._add_module_to_engine:
            logger.info(f"Adding a datasource to the engine (key: {ct.DATA_SOURCE})...")
            engine.add_module(ct.DATA_SOURCE, datasource)
        return datasource
