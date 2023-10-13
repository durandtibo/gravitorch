from __future__ import annotations

__all__ = ["DataPipeDataSource", "DataCreatorDataSource"]

import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, TypeVar

from coola import summary
from coola.utils import str_indent, str_mapping
from torch.utils.data import IterDataPipe, MapDataPipe

from gravitorch.creators.datapipe.base import (
    BaseDataPipeCreator,
    setup_datapipe_creator,
)
from gravitorch.data.datacreators.base import BaseDataCreator, setup_datacreator
from gravitorch.datasources.base import BaseDataSource, IterableNotFoundError
from gravitorch.utils.asset import AssetManager

if TYPE_CHECKING:
    from gravitorch.engines import BaseEngine

logger = logging.getLogger(__name__)

T = TypeVar("T")


class DataPipeDataSource(BaseDataSource):
    r"""Implements a datasource that creates data loaders using
    ``DataPipe`` creators.

    Args:
    ----
        datapipe_creators (dict): Specifies the ``DataPipe``
            creators. Each key is associated to a datastream ID.
            For example if you want to use a ``'train'`` datastream,
            you need to have a key associated to a
            ``BaseDataPipeCreator`` object or its configuration.
            Each ``BaseDataPipeCreator`` object contains the
            recipe to create a ``DataPipe`` object.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.datasources import DataPipeDataSource
        >>> from gravitorch.creators.datapipe import ChainedDataPipeCreator
        >>> datasource = DataPipeDataSource(
        ...     datapipe_creators={
        ...         "train": ChainedDataPipeCreator(
        ...             config=[
        ...                 {
        ...                     "_target_": "torch.utils.data.datapipes.iter.IterableWrapper",
        ...                     "iterable": [1, 2, 3, 4],
        ...                 },
        ...             ]
        ...         ),
        ...         "val": ChainedDataPipeCreator(
        ...             config=[
        ...                 {
        ...                     "_target_": "torch.utils.data.datapipes.iter.IterableWrapper",
        ...                     "iterable": ["a", "b", "c"],
        ...                 },
        ...             ]
        ...         ),
        ...     }
        ... )
        >>> datasource
        DataPipeDataSource(
          (train): ChainedDataPipeCreator(
              (0): {'_target_': 'torch.utils.data.datapipes.iter.IterableWrapper', 'iterable': [1, 2, 3, 4]}
            )
          (val): ChainedDataPipeCreator(
              (0): {'_target_': 'torch.utils.data.datapipes.iter.IterableWrapper', 'iterable': ['a', 'b', 'c']}
            )
        )
        >>> # Create by using the configs
        >>> # Note that both examples lead to the same result.
        >>> datasource = DataPipeDataSource(
        ...     datapipe_creators={
        ...         "train": {
        ...             "_target_": "gravitorch.creators.datapipe.ChainedDataPipeCreator",
        ...             "config": [
        ...                 {
        ...                     "_target_": "torch.utils.data.datapipes.iter.IterableWrapper",
        ...                     "iterable": [1, 2, 3, 4],
        ...                 },
        ...             ],
        ...         },
        ...         "val": {
        ...             "_target_": "gravitorch.creators.datapipe.ChainedDataPipeCreator",
        ...             "config": [
        ...                 {
        ...                     "_target_": "torch.utils.data.datapipes.iter.IterableWrapper",
        ...                     "iterable": ["a", "b", "c"],
        ...                 },
        ...             ],
        ...         },
        ...     }
        ... )
        >>> datasource
        DataPipeDataSource(
          (train): ChainedDataPipeCreator(
              (0): {'_target_': 'torch.utils.data.datapipes.iter.IterableWrapper', 'iterable': [1, 2, 3, 4]}
            )
          (val): ChainedDataPipeCreator(
              (0): {'_target_': 'torch.utils.data.datapipes.iter.IterableWrapper', 'iterable': ['a', 'b', 'c']}
            )
        )
    """

    def __init__(self, datapipe_creators: dict[str, BaseDataPipeCreator | dict]) -> None:
        self._asset_manager = AssetManager()
        logger.info("Initializing the DataPipe creators...")
        self._datapipe_creators = {
            key: setup_datapipe_creator(creator) for key, creator in datapipe_creators.items()
        }
        logger.info(f"DataPipe creators:\n{str_mapping(self._datapipe_creators)}")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  {str_indent(str_mapping(self._datapipe_creators))}\n)"
        )

    def attach(self, engine: BaseEngine) -> None:
        logger.info("Attach the datasource to an engine")

    def get_asset(self, asset_id: str) -> Any:
        return self._asset_manager.get_asset(asset_id)

    def has_asset(self, asset_id: str) -> bool:
        return self._asset_manager.has_asset(asset_id)

    def get_iterable(self, iter_id: str, engine: BaseEngine | None = None) -> Iterable[T]:
        if not self.has_iterable(iter_id):
            raise IterableNotFoundError(f"{iter_id} does not exist")
        return self._create_datapipe(loader_id=iter_id, engine=engine)

    def has_iterable(self, iter_id: str) -> bool:
        return iter_id in self._datapipe_creators

    def _create_datapipe(
        self, loader_id: str, engine: BaseEngine | None = None
    ) -> IterDataPipe[T] | MapDataPipe[T]:
        r"""Creates an ``DataPipe`` object.

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
            ``IterDataPipe`` or ``MapDataPipe``: A DataPipe object.
        """
        logger.info("Crating DataPipe...")
        datapipe = self._datapipe_creators[loader_id].create(engine=engine)
        logger.info(f"Created DataPipe:\n{datapipe}")
        return datapipe


class DataCreatorDataSource(DataPipeDataSource):
    r"""Implements a datasource that creates data loaders using
    ``DataPipe`` creators.

    Unlike ``DataPipeDataSource``, each ``DataPipe``
    creator takes as input (``source_inputs``) the data created by a
    ``BaseDataCreator`` object if it is defined. If no
    ``BaseDataCreator`` object is defined, ``source_inputs`` of the
    ``DataPipe`` creator is set to ``None``.

    Args:
    ----
        datapipe_creators (dict): Specifies the ``DataPipe``
            creators or their configurations. Each key is associated
            to a loader ID. For example if you want to use a
            ``'train'`` data loader, you need to map this key to a
            ``BaseDataPipeCreator`` object or its configuration.
            Each ``BaseDataPipeCreator`` object contains the
            recipe to create an ``DataPipe`` object.
        data_creators (dict): Specifies the data creators or their
            configurations. Each key is associated to a loader ID.
            For example if you want to create data for the ``'train'``
            loader, you need to map this key to a ``BaseDataCreator``
            object or its configuration.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.datasources import DataCreatorDataSource
        >>> from gravitorch.creators.datapipe import ChainedDataPipeCreator
        >>> datasource = DataCreatorDataSource(
        ...     datapipe_creators={
        ...         "train": ChainedDataPipeCreator(
        ...             config=[
        ...                 {
        ...                     "_target_": "torch.utils.data.datapipes.iter.IterableWrapper",
        ...                     "iterable": [1, 2, 3, 4],
        ...                 },
        ...             ]
        ...         ),
        ...         "val": ChainedDataPipeCreator(
        ...             config=[
        ...                 {
        ...                     "_target_": "torch.utils.data.datapipes.iter.IterableWrapper",
        ...                     "iterable": ["a", "b", "c"],
        ...                 },
        ...             ]
        ...         ),
        ...     },
        ...     data_creators={
        ...         "train": {
        ...             "_target_": "gravitorch.data.datacreators.HypercubeVertexDataCreator",
        ...             "num_examples": 10,
        ...             "num_classes": 5,
        ...         },
        ...         "val": {
        ...             "_target_": "gravitorch.data.datacreators.HypercubeVertexDataCreator",
        ...             "num_examples": 10,
        ...             "num_classes": 5,
        ...         },
        ...     },
        ... )
        >>> datasource
        DataCreatorDataSource(
          (data_creators):
            (train): HypercubeVertexDataCreator(num_examples=10, num_classes=5, feature_size=64, noise_std=0.2, random_seed=15782179921860610490)
            (val): HypercubeVertexDataCreator(num_examples=10, num_classes=5, feature_size=64, noise_std=0.2, random_seed=15782179921860610490)
          (datapipe_creators):
            (train): ChainedDataPipeCreator(
                (0): {'_target_': 'torch.utils.data.datapipes.iter.IterableWrapper', 'iterable': [1, 2, 3, 4]}
              )
            (val): ChainedDataPipeCreator(
                (0): {'_target_': 'torch.utils.data.datapipes.iter.IterableWrapper', 'iterable': ['a', 'b', 'c']}
              )
        )
    """

    def __init__(
        self,
        datapipe_creators: dict[str, BaseDataPipeCreator | dict],
        data_creators: dict[str, BaseDataCreator | dict],
    ) -> None:
        super().__init__(datapipe_creators)
        logger.info("Initializing the data creators...")
        self._data_creators = {
            key: setup_datacreator(creator) for key, creator in data_creators.items()
        }
        logger.info(f"Data creators:\n{str_mapping(self._data_creators)}")
        logger.info("Creating data...")
        self._data = {key: creator.create() for key, creator in self._data_creators.items()}
        logger.info(f"Data:\n{summary(self._data, max_depth=2)}")

    def __repr__(self) -> str:
        args = str_indent(
            str_mapping(
                {
                    "data_creators": "\n" + str_mapping(self._data_creators)
                    if self._data_creators
                    else "",
                    "datapipe_creators": "\n" + str_mapping(self._datapipe_creators)
                    if self._datapipe_creators
                    else "",
                }
            )
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def _create_datapipe(
        self, loader_id: str, engine: BaseEngine | None = None
    ) -> IterDataPipe | MapDataPipe:
        r"""Creates an ``DataPipe`` object.

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
            ``IterDataPipe`` or ``MapDataPipe``: A ``DataPipe`` object.
        """
        source_input = self._data.get(loader_id, None)
        return self._datapipe_creators[loader_id].create(
            engine=engine,
            source_inputs=source_input if source_input is None else (source_input,),
        )
