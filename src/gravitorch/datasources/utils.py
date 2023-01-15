__all__ = ["setup_data_source", "setup_and_attach_data_source"]

import logging
from typing import Union

from gravitorch.datasources.base import BaseDataSource
from gravitorch.engines.base import BaseEngine
from gravitorch.utils.format import str_target_object

logger = logging.getLogger(__name__)


def setup_data_source(data_source: Union[BaseDataSource, dict]) -> BaseDataSource:
    r"""Sets up a data source.

    The data source is instantiated from its configuration by using
    the ``BaseDataSource`` factory function.

    Args:
        data_source (``BaseDataSource`` or dict): Specifies the data
            source or its configuration.

    Returns:
        ``BaseDataSource``: The instantiated data source.
    """
    if isinstance(data_source, dict):
        logger.info(
            "Initializing a data source from its configuration... "
            f"{str_target_object(data_source)}"
        )
        data_source = BaseDataSource.factory(**data_source)
    return data_source


def setup_and_attach_data_source(
    data_source: Union[BaseDataSource, dict], engine: BaseEngine
) -> BaseDataSource:
    r"""Sets up a data source and attach it to an engine.

    Note that if you call this function ``N`` times with the same data
    source object, the data source will be attached ``N`` times to the
    engine.

    Args:
        data_source (``BaseDataSource`` or dict): Specifies the data
            source or its configuration.
        engine (``BaseEngine``): Specifies the engine.

    Returns:
        ``BaseDataSource``: The instantiated data source.
    """
    data_source = setup_data_source(data_source)
    logger.info("Adding a data source object to an engine...")
    data_source.attach(engine)
    return data_source
