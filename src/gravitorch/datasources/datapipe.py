__all__ = ["IterDataPipeCreatorDataSource", "DataCreatorIterDataPipeCreatorDataSource"]

import logging
from collections.abc import Iterable
from typing import Any, Optional, TypeVar, Union

from torch.utils.data import IterDataPipe

from gravitorch.creators.datapipe import (
    BaseIterDataPipeCreator,
    setup_iter_datapipe_creator,
)
from gravitorch.data.datacreators import BaseDataCreator, setup_data_creator
from gravitorch.datasources.base import BaseDataSource, LoaderNotFoundError
from gravitorch.engines.base import BaseEngine
from gravitorch.utils.asset import AssetManager
from gravitorch.utils.format import str_indent, to_torch_mapping_str
from gravitorch.utils.summary import concise_summary

logger = logging.getLogger(__name__)

T = TypeVar("T")


class IterDataPipeCreatorDataSource(BaseDataSource):
    r"""Implements a data source that creates data loaders using
    ``IterDataPipe`` creators.

    Args:
        datapipe_creators (dict): Specifies the ``IterDataPipe``
            creators. Each key is associated to a loader ID. For
            example if you want to use a ``'train'`` data loader,
            you need to have a key associated to a
            ``BaseIterDataPipeCreator`` object or its configuration.
            Each ``BaseIterDataPipeCreator`` object contains the
            recipe to create an ``IterDataPipe`` object.

    Example:

    .. code-block:: python

        >>> from gravitorch.datasources import IterDataPipeCreatorDataSource
        # Create by using a ``BaseIterDataPipeCreator`` object.
        >>> from gravitorch.creators.datapipe import SequentialIterDataPipeCreator
        >>> from gravitorch.data.datapipes.iter import SourceWrapper
        >>> data_source = IterDataPipeCreatorDataSource(
        ...     datapipe_creators={
        ...         "train": SequentialIterDataPipeCreator(
        ...             config=[
        ...                 {
        ...                     '_target_': "gravitorch.data.datapipes.iter.SourceWrapper",
        ...                     "data": [1, 2, 3, 4],
        ...                 },
        ...             ]
        ...         ),
        ...         "val": SequentialIterDataPipeCreator(
        ...             config=[
        ...                 {
        ...                     '_target_': "gravitorch.data.datapipes.iter.SourceWrapper",
        ...                     "data": ["a", "b", "c"],
        ...                 },
        ...             ]
        ...         ),
        ...     }
        ... )
        # Create by using the configs
        >>> from objectory import OBJECT_TARGET
        >>> data_source = IterDataPipeCreatorDataSource(
        ...     datapipe_creators={
        ...         "train": {
        ...             '_target_': "gravitorch.creators.datapipe.SequentialIterDataPipeCreator",
        ...             "config": [
        ...                 {
        ...                     OBJECT_TARGET: "gravitorch.data.datapipes.iter.SourceWrapper",
        ...                    "data": [1, 2, 3, 4],
        ...                },
        ...             ],
        ...         },
        ...         "val": {
        ...             '_target_': "gravitorch.creators.datapipe.SequentialIterDataPipeCreator",
        ...             "config": [
        ...                 {
        ...                     '_target_': "gravitorch.data.datapipes.iter.SourceWrapper",
        ...                     "data": ["a", "b", "c"],
        ...                 },
        ...             ],
        ...         },
        ...     }
        ... )
        # Note that both examples lead to the same result.
    """

    def __init__(self, datapipe_creators: dict[str, Union[BaseIterDataPipeCreator, dict]]):
        self._asset_manager = AssetManager()
        logger.info("Initializing the IterDataPipe creators...")
        self._datapipe_creators = {
            key: setup_iter_datapipe_creator(creator) for key, creator in datapipe_creators.items()
        }
        logger.info(f"IterDataPipe creators:\n{to_torch_mapping_str(self._datapipe_creators)}")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  {str_indent(to_torch_mapping_str(self._datapipe_creators))}\n)"
        )

    def attach(self, engine: BaseEngine) -> None:
        r"""Attaches the current data source to the provided engine.

        This method can be used to set up events or logs some stats
        to the engine.

        Args:
            engine (``BaseEngine``): Specifies the engine.

        Example:

        .. code-block:: python

            >>> from gravitorch.datasources import IterDataPipeCreatorDataSource
            >>> data_source = IterDataPipeCreatorDataSource(datapipe_creators={...})
            >>> from gravitorch.engines import AlphaEngine
            >>> my_engine = AlphaEngine()  # Work with any engine
            >>> data_source.attach(my_engine)
        """
        logger.info("Attach the data source to an engine")

    def get_asset(self, asset_id: str) -> Any:
        r"""Gets a data asset from this data source.

        This method is useful to access some data variables/parameters
        that are not available before to load/preprocess the data.

        Args:
            asset_id (str): Specifies the ID of the asset.

        Returns:
            The asset.

        Raises:
            ``AssetNotFoundError`` if you try to access an asset
                that does not exist.

        Example:

        .. code-block:: python

            >>> from gravitorch.datasources import IterDataPipeCreatorDataSource
            >>> data_source = IterDataPipeCreatorDataSource(datapipe_creators={...})
            >>> my_asset = data_source.get_asset('my_asset_id')
        """
        return self._asset_manager.get_asset(asset_id)

    def has_asset(self, asset_id: str) -> bool:
        r"""Indicates if the asset exists or not.

        Args:
            asset_id (str): Specifies the ID of the asset.

        Returns:
            bool: ``True`` if the asset exists, otherwise ``False``.

        Example:

        .. code-block:: python

            >>> from gravitorch.datasources import IterDataPipeCreatorDataSource
            >>> data_source = IterDataPipeCreatorDataSource(datapipe_creators={...})
            >>> data_source.has_asset('my_asset_id')
            False
        """
        return self._asset_manager.has_asset(asset_id)

    def get_data_loader(self, loader_id: str, engine: Optional[BaseEngine] = None) -> Iterable[T]:
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

            >>> from gravitorch.datasources import IterDataPipeCreatorDataSource
            >>> data_source = IterDataPipeCreatorDataSource(datapipe_creators={...})
            # Get the data loader associated to the ID 'train'
            >>> data_loader = data_source.get_data_loader('train')
            # Get a data loader that can use information from an engine
            >>> from gravitorch.engines import AlphaEngine
            >>> my_engine = AlphaEngine()  # Work with any engine
            >>> data_loader = data_source.get_data_loader('train', my_engine)
        """
        if not self.has_data_loader(loader_id):
            raise LoaderNotFoundError(f"{loader_id} does not exist")
        return self._create_datapipe(loader_id=loader_id, engine=engine)

    def has_data_loader(self, loader_id: str) -> bool:
        r"""Indicates if the data source has a data loader with the given ID.

        Args:
            loader_id (str): Specifies the ID of the data loader.

        Returns:
            bool: ``True`` if the data loader exists, ``False``
                otherwise.

        Example:

        .. code-block:: python

            >>> from gravitorch.datasources import IterDataPipeCreatorDataSource
            >>> data_source = IterDataPipeCreatorDataSource(datapipe_creators={...})
            # Check if the data source has a data loader for ID 'train'
            >>> data_source.has_data_loader('train')
            True or False
            # Check if the data source has a data loader for ID 'eval'
            >>> data_source.has_data_loader('eval')
            True or False
        """
        return loader_id in self._datapipe_creators

    def _create_datapipe(
        self, loader_id: str, engine: Optional[BaseEngine] = None
    ) -> IterDataPipe[T]:
        r"""Creates an ``IterDataPipe`` object.

        Args:
            loader_id (str): Specifies the ID of the data loader to
                get.
            engine (``BaseEngine`` or ``None``, optional): Specifies
                an engine. The engine can be used to initialize the
                data loader by using the current epoch value.
                Default: ``None``

        Returns:
            ``IterDataPipe``: An ``IterDataPipe`` object.
        """
        logger.info("Crating DataPipe...")
        datapipe = self._datapipe_creators[loader_id].create(engine=engine)
        logger.info(f"Created DataPipe:\n{datapipe}")
        return datapipe


class DataCreatorIterDataPipeCreatorDataSource(IterDataPipeCreatorDataSource):
    r"""Implements a data source that creates data loaders using
    ``IterDataPipe`` creators.

    Unlike ``IterDataPipeCreatorDataSource``, each ``IterDataPipe``
    creator takes as input (``source_inputs``) the data created by a
    ``BaseDataCreator`` object if it is defined. If no
    ``BaseDataCreator`` object is defined, ``source_inputs`` of the
    ``IterDataPipe`` creator is set to ``None``.

    Args:
        datapipe_creators (dict): Specifies the ``IterDataPipe``
            creators or their configurations. Each key is associated
            to a loader ID. For example if you want to use a
            ``'train'`` data loader, you need to map this key to a
            ``BaseIterDataPipeCreator`` object or its configuration.
            Each ``BaseIterDataPipeCreator`` object contains the
            recipe to create an ``IterDataPipe`` object.
        data_creators (dict): Specifies the data creators or their
            configurations. Each key is associated to a loader ID.
            For example if you want to create data for the ``'train'``
            loader, you need to map this key to a ``BaseDataCreator``
            object or its configuration.
    """

    def __init__(
        self,
        datapipe_creators: dict[str, Union[BaseIterDataPipeCreator, dict]],
        data_creators: dict[str, Union[BaseDataCreator, dict]],
    ):
        super().__init__(datapipe_creators)
        logger.info("Initializing the data creators...")
        self._data_creators = {
            key: setup_data_creator(creator) for key, creator in data_creators.items()
        }
        logger.info(f"Data creators:\n{to_torch_mapping_str(self._data_creators)}")
        logger.info("Creating data...")
        self._data = {key: creator.create() for key, creator in self._data_creators.items()}
        logger.info(f"Data:\n{concise_summary(self._data)}")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(\n"
            "  data_creators\n"
            f"    {str_indent(to_torch_mapping_str(self._data_creators), num_spaces=4)}\n"
            "  datapipe_creators\n"
            f"    {str_indent(to_torch_mapping_str(self._datapipe_creators), num_spaces=4)}"
            "\n)"
        )

    def _create_datapipe(self, loader_id: str, engine: Optional[BaseEngine] = None) -> IterDataPipe:
        r"""Creates an ``IterDataPipe`` object.

        Args:
            loader_id (str): Specifies the ID of the data loader to
                get.
            engine (``BaseEngine`` or ``None``, optional): Specifies
                an engine. The engine can be used to initialize the
                data loader by using the current epoch value.
                Default: ``None``

        Returns:
            ``IterDataPipe``: An ``IterDataPipe`` object.
        """
        source_input = self._data.get(loader_id, None)
        return self._datapipe_creators[loader_id].create(
            engine=engine,
            source_inputs=source_input if source_input is None else (source_input,),
        )
