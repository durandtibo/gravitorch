from __future__ import annotations

__all__ = [
    "BaseDataSource",
    "LoaderNotFoundError",
    "is_datasource_config",
    "setup_datasource",
    "setup_and_attach_datasource",
]

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from objectory import AbstractFactory
from objectory.utils import is_object_config

from gravitorch.utils.format import str_target_object

if TYPE_CHECKING:
    from gravitorch.engines import BaseEngine

logger = logging.getLogger(__name__)

T = TypeVar("T")


class BaseDataSource(ABC, Generic[T], metaclass=AbstractFactory):
    r"""Defines the base class to implement a datasource.

    A datasource object is responsible to create the data loaders.

    Note: it is an experimental class and the API may change.
    """

    @abstractmethod
    def attach(self, engine: BaseEngine) -> None:
        r"""Attaches the current datasource to the provided engine.

        This method can be used to set up events or logs some stats to
        the engine.

        Args:
        ----
            engine (``BaseEngine``): Specifies the engine.

        Example:
        -------
        .. code-block:: pycon

            >>> from gravitorch.datasources import BaseDataSource
            >>> from gravitorch.engines import AlphaEngine
            >>> my_engine = AlphaEngine()  # Work with any engine
            >>> datasource: BaseDataSource = ...  # Instantiate a datasource
            >>> datasource.attach(my_engine)
        """

    @abstractmethod
    def get_asset(self, asset_id: str) -> Any:
        r"""Gets a data asset from this datasource.

        This method is useful to access some data variables/parameters
        that are not available before to load/preprocess the data.

        Args:
        ----
            asset_id (str): Specifies the ID of the asset.

        Returns:
        -------
            The asset.

        Raises:
        ------
            ``AssetNotFoundError`` if you try to access an asset that
                does not exist.

        Example:
        -------
        .. code-block:: pycon

            >>> from gravitorch.datasources import BaseDataSource
            >>> datasource: BaseDataSource = ...  # Instantiate a datasource
            >>> my_asset = datasource.get_asset("my_asset_id")
        """

    @abstractmethod
    def has_asset(self, asset_id: str) -> bool:
        r"""Indicates if the asset exists or not.

        Args:
        ----
            asset_id (str): Specifies the ID of the asset.

        Returns:
        -------
            bool: ``True`` if the asset exists, otherwise ``False``.

        Example:
        -------
        .. code-block:: pycon

            >>> from gravitorch.datasources import BaseDataSource
            >>> datasource: BaseDataSource = ...  # Instantiate a datasource
            >>> datasource.has_asset("my_asset_id")
            False
        """

    @abstractmethod
    def get_data_loader(self, loader_id: str, engine: BaseEngine | None = None) -> Iterable[T]:
        r"""Gets a data loader.

        Args:
        ----
            loader_id (str): Specifies the ID of the data loader to
                get.
            engine (``BaseEngine`` or ``None``, optional): Specifies
                an engine. The engine can be used to initialize the
                data loader by using the current epoch value.
                Default: ``None``

        Returns:
        -------
            ``Iterable``: A data loader.

        Raises:
        ------
            ``LoaderNotFoundError`` if the loader does not exist.

        Example:
        -------
        .. code-block:: pycon

            >>> from gravitorch.datasources import BaseDataSource
            >>> datasource: BaseDataSource = ...  # Instantiate a datasource
            # Get the data loader associated to the ID 'train'
            >>> data_loader = datasource.get_data_loader("train")
            # Get a data loader that can use information from an engine
            >>> from gravitorch.engines import AlphaEngine
            >>> my_engine = AlphaEngine()  # Work with any engine
            >>> data_loader = datasource.get_data_loader("train", my_engine)
        """

    @abstractmethod
    def has_data_loader(self, loader_id: str) -> bool:
        r"""Indicates if the datasource has a data loader with the given
        ID.

        Args:
        ----
            loader_id (str): Specifies the ID of the data loader.

        Returns:
        -------
            bool: ``True`` if the data loader exists, ``False``
                otherwise.

        Example:
        -------
        .. code-block:: pycon

            >>> from gravitorch.datasources import BaseDataSource
            >>> datasource: BaseDataSource = ...  # Instantiate a datasource
            # Check if the datasource has a data loader for ID 'train'
            >>> datasource.has_data_loader("train")
            True or False
            # Check if the datasource has a data loader for ID 'eval'
            >>> datasource.has_data_loader("eval")
            True or False
        """

    def load_state_dict(self, state_dict: dict) -> None:
        r"""Loads the state values from a dict.

        Args:
        ----
            state_dict (dict): a dict with parameters

        Example:
        -------
        .. code-block:: pycon

            >>> from gravitorch.datasources import BaseDataSource
            >>> datasource: BaseDataSource = ...  # Instantiate a datasource
            # Please take a look to the implementation of the state_dict
            # function to know the expected structure
            >>> state_dict = {...}
            >>> datasource.load_state_dict(state_dict)
        """

    def state_dict(self) -> dict:
        r"""Returns a dictionary containing state values.

        Returns:
        -------
            dict: the state values in a dict.

        Example:
        -------
        .. code-block:: pycon

            >>> from gravitorch.datasources import BaseDataSource
            >>> datasource: BaseDataSource = ...  # Instantiate a datasource
            >>> state_dict = datasource.state_dict()
            {...}
        """
        return {}


class LoaderNotFoundError(Exception):
    r"""Raised when a loader is requested but does not exist."""


def is_datasource_config(config: dict) -> bool:
    r"""Indicate if the input configuration is a configuration for a
    ``BaseDataSource``.

    This function only checks if the value of the key  ``_target_``
    is valid. It does not check the other values.

    Args:
    ----
        config (dict): Specifies the configuration to check.

    Returns:
    -------
        bool: ``True`` if the input configuration is a configuration
            for a ``BaseDataSource`` object.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.datasources import is_datasource_config
        >>> is_datasource_config(
        ...     {"_target_": "gravitorch.datasources.IterDataPipeCreatorDataSource"}
        ... )
        True
    """
    return is_object_config(config, BaseDataSource)


def setup_datasource(datasource: BaseDataSource | dict) -> BaseDataSource:
    r"""Sets up a datasource.

    The datasource is instantiated from its configuration by using
    the ``BaseDataSource`` factory function.

    Args:
    ----
        datasource (``BaseDataSource`` or dict): Specifies the data
            source or its configuration.

    Returns:
    -------
        ``BaseDataSource``: The instantiated datasource.
    """
    if isinstance(datasource, dict):
        logger.info(
            "Initializing a datasource from its configuration... "
            f"{str_target_object(datasource)}"
        )
        datasource = BaseDataSource.factory(**datasource)
    return datasource


def setup_and_attach_datasource(
    datasource: BaseDataSource | dict, engine: BaseEngine
) -> BaseDataSource:
    r"""Sets up a datasource and attach it to an engine.

    Note that if you call this function ``N`` times with the same data
    source object, the datasource will be attached ``N`` times to the
    engine.

    Args:
    ----
        datasource (``BaseDataSource`` or dict): Specifies the data
            source or its configuration.
        engine (``BaseEngine``): Specifies the engine.

    Returns:
    -------
        ``BaseDataSource``: The instantiated datasource.
    """
    datasource = setup_datasource(datasource)
    logger.info("Adding a datasource object to an engine...")
    datasource.attach(engine)
    return datasource
