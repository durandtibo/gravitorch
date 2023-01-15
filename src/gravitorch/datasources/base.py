from __future__ import annotations

__all__ = ["BaseDataSource", "LoaderNotFoundError"]

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from objectory import AbstractFactory

if TYPE_CHECKING:
    from gravitorch.engines import BaseEngine

logger = logging.getLogger(__name__)

T = TypeVar("T")


class BaseDataSource(ABC, Generic[T], metaclass=AbstractFactory):
    r"""Defines the base class to implement a data source.

    A data source object is responsible to create the data loaders.

    Note: it is an experimental class and the API may change.
    """

    @abstractmethod
    def attach(self, engine: BaseEngine) -> None:
        r"""Attaches the current data source to the provided engine.

        This method can be used to set up events or logs some stats to
        the engine.

        Args:
            engine (``BaseEngine``): Specifies the engine.

        Example:

        .. code-block:: python

            >>> from gravitorch.datasources import BaseDataSource
            >>> from gravitorch.engines import AlphaEngine
            >>> my_engine = AlphaEngine()  # Work with any engine
            >>> data_source: BaseDataSource = ...  # Instantiate a data source
            >>> data_source.attach(my_engine)
        """

    @abstractmethod
    def get_asset(self, asset_id: str) -> Any:
        r"""Gets a data asset from this data source.

        This method is useful to access some data variables/parameters
        that are not available before to load/preprocess the data.

        Args:
            asset_id (str): Specifies the ID of the asset.

        Returns:
            The asset.

        Raises:
            ``AssetNotFoundError`` if you try to access an asset that
                does not exist.

        Example:

        .. code-block:: python

            >>> from gravitorch.datasources import BaseDataSource
            >>> data_source: BaseDataSource = ...  # Instantiate a data source
            >>> my_asset = data_source.get_asset('my_asset_id')
        """

    @abstractmethod
    def has_asset(self, asset_id: str) -> bool:
        r"""Indicates if the asset exists or not.

        Args:
            asset_id (str): Specifies the ID of the asset.

        Returns:
            bool: ``True`` if the asset exists, otherwise ``False``.

        Example:

        .. code-block:: python

            >>> from gravitorch.datasources import BaseDataSource
            >>> data_source: BaseDataSource = ...  # Instantiate a data source
            >>> data_source.has_asset('my_asset_id')
            False
        """

    @abstractmethod
    def get_data_loader(self, loader_id: str, engine: BaseEngine | None = None) -> Iterable[T]:
        r"""Gets a data loader.

        Args:
            loader_id (str): Specifies the ID of the data loader to
                get.
            engine (``BaseEngine`` or ``None``, optional): Specifies
                an engine. The engine can be used to initialize the
                data loader by using the current epoch value.
                Default: ``None``

        Returns:
            ``Iterable``: A data loader.

        Raises:
            ``LoaderNotFoundError`` if the loader does not exist.

        Example:

        .. code-block:: python

            >>> from gravitorch.datasources import BaseDataSource
            >>> data_source: BaseDataSource = ...  # Instantiate a data source
            # Get the data loader associated to the ID 'train'
            >>> data_loader = data_source.get_data_loader('train')
            # Get a data loader that can use information from an engine
            >>> from gravitorch.engines import AlphaEngine
            >>> my_engine = AlphaEngine()  # Work with any engine
            >>> data_loader = data_source.get_data_loader('train', my_engine)
        """

    @abstractmethod
    def has_data_loader(self, loader_id: str) -> bool:
        r"""Indicates if the data source has a data loader with the given ID.

        Args:
            loader_id (str): Specifies the ID of the data loader.

        Returns:
            bool: ``True`` if the data loader exists, ``False``
                otherwise.

        Example:

        .. code-block:: python

            >>> from gravitorch.datasources import BaseDataSource
            >>> data_source: BaseDataSource = ...  # Instantiate a data source
            # Check if the data source has a data loader for ID 'train'
            >>> data_source.has_data_loader('train')
            True or False
            # Check if the data source has a data loader for ID 'eval'
            >>> data_source.has_data_loader('eval')
            True or False
        """

    def load_state_dict(self, state_dict: dict) -> None:
        r"""Loads the state values from a dict.

        Args:
            state_dict (dict): a dict with parameters

        Example:

        .. code-block:: python

            >>> from gravitorch.datasources import BaseDataSource
            >>> data_source: BaseDataSource = ...  # Instantiate a data source
            # Please take a look to the implementation of the state_dict
            # function to know the expected structure
            >>> state_dict = {...}
            >>> data_source.load_state_dict(state_dict)
        """

    def state_dict(self) -> dict:
        r"""Returns a dictionary containing state values.

        Returns:
            dict: the state values in a dict.

        Example:

        .. code-block:: python

            >>> from gravitorch.datasources import BaseDataSource
            >>> data_source: BaseDataSource = ...  # Instantiate a data source
            >>> state_dict = data_source.state_dict()
            {...}
        """
        return {}


class LoaderNotFoundError(Exception):
    r"""Raised when a loader is requested but does not exist."""
