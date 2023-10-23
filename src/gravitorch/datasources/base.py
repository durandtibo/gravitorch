from __future__ import annotations

__all__ = [
    "BaseDataSource",
    "IterableNotFoundError",
    "is_datasource_config",
    "setup_and_attach_datasource",
    "setup_datasource",
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

    A datasource object is responsible to create the data iterables.

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

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.testing import DummyDataSource, create_dummy_engine
            >>> engine = create_dummy_engine()
            >>> datasource = DummyDataSource()
            >>> datasource.attach(engine)
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

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.testing import DummyDataSource, create_dummy_engine
            >>> engine = create_dummy_engine()
            >>> datasource = DummyDataSource()
            >>> train_dataset = datasource.get_asset("train_dataset")
            >>> train_dataset
            DummyDataset(num_examples=4, feature_size=4)
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

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.testing import DummyDataSource, create_dummy_engine
            >>> engine = create_dummy_engine()
            >>> datasource = DummyDataSource()
            >>> datasource.has_asset("missing")
            False
        """

    @abstractmethod
    def get_iterable(self, iter_id: str, engine: BaseEngine | None = None) -> Iterable[T]:
        r"""Gets a data iterable for the given ID.

        Args:
        ----
            iter_id (str): Specifies the ID of the data iterable
                to get.
            engine (``BaseEngine`` or ``None``, optional): Specifies
                an engine. The engine can be used to initialize the
                data iterable by using the current epoch value.
                Default: ``None``

        Returns:
        -------
            ``Iterable`` : A data iterable.

        Raises:
        ------
            ``IterableNotFoundError`` if the loader does not exist.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.testing import DummyDataSource, create_dummy_engine
            >>> datasource = DummyDataSource()
            >>> dataiter = datasource.get_iterable("train")
            >>> dataiter
            <torch.utils.data.dataloader.DataLoader object at 0x...>
            >>> # Get a iterable that can use information from an engine
            >>> engine = create_dummy_engine()
            >>> dataiter = datasource.get_iterable("train", engine)
            >>> dataiter
            <torch.utils.data.dataloader.DataLoader object at 0x...>
        """

    @abstractmethod
    def has_iterable(self, iter_id: str) -> bool:
        r"""Indicates if the datasource has a data iterable for the given
        ID.

        Args:
        ----
            iter_id (str): Specifies the ID of the data iterable.

        Returns:
        -------
            bool: ``True`` if the data iterable exists, ``False``
                otherwise.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.testing import DummyDataSource
            >>> datasource = DummyDataSource()
            >>> datasource.has_iterable("train")
            True
            >>> datasource.has_iterable("eval")
            True
            >>> datasource.has_iterable("missing")
            False
        """

    def load_state_dict(self, state_dict: dict) -> None:
        r"""Loads the state values from a dict.

        Args:
        ----
            state_dict (dict): a dict with parameters

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.testing import DummyDataSource
            >>> datasource = DummyDataSource()
            >>> # Please take a look to the implementation of the state_dict
            >>> # function to know the expected structure
            >>> datasource.load_state_dict({})
        """

    def state_dict(self) -> dict:
        r"""Returns a dictionary containing state values.

        Returns:
        -------
            dict: the state values in a dict.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.testing import DummyDataSource
            >>> datasource = DummyDataSource()
            >>> state = datasource.state_dict()
            >>> state
            {}
        """
        return {}


class IterableNotFoundError(Exception):
    r"""Raised when a data iterable is requires but does not exist."""


def is_datasource_config(config: dict) -> bool:
    r"""Indicates if the input configuration is a configuration for a
    ``BaseDataSource``.

    This function only checks if the value of the key  ``_target_``
    is valid. It does not check the other values. If ``_target_``
    indicates a function, the returned type hint is used to check
    the class.

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
        >>> is_datasource_config({"_target_": "gravitorch.datasources.DataPipeDataSource"})
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

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.datasources import setup_datasource
        >>> datasource = setup_datasource({"_target_": "gravitorch.testing.DummyDataSource"})
        >>> datasource
        DummyDataSource(
          (datasets):
            (train): DummyDataset(num_examples=4, feature_size=4)
            (eval): DummyDataset(num_examples=4, feature_size=4)
          (dataloader_creators):
            (train): DataLoaderCreator(
                (seed): 0
                (batch_size): 1
                (shuffle): False
              )
            (eval): DataLoaderCreator(
                (seed): 0
                (batch_size): 1
                (shuffle): False
              )
        )
    """
    if isinstance(datasource, dict):
        logger.info(
            "Initializing a datasource from its configuration... "
            f"{str_target_object(datasource)}"
        )
        datasource = BaseDataSource.factory(**datasource)
    if not isinstance(datasource, BaseDataSource):
        logger.warning(f"datasource is not a `BaseDataSource` (received: {type(datasource)})")
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

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.datasources import setup_and_attach_datasource
        >>> from gravitorch.testing import create_dummy_engine
        >>> engine = create_dummy_engine()
        >>> datasource = setup_and_attach_datasource(
        ...     {"_target_": "gravitorch.testing.DummyDataSource"}, engine=engine
        ... )
        >>> datasource
        DummyDataSource(
          (datasets):
            (train): DummyDataset(num_examples=4, feature_size=4)
            (eval): DummyDataset(num_examples=4, feature_size=4)
          (dataloader_creators):
            (train): DataLoaderCreator(
                (seed): 0
                (batch_size): 1
                (shuffle): False
              )
            (eval): DataLoaderCreator(
                (seed): 0
                (batch_size): 1
                (shuffle): False
              )
        )
    """
    datasource = setup_datasource(datasource)
    logger.info("Adding a datasource object to an engine...")
    datasource.attach(engine)
    return datasource
