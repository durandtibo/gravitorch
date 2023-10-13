from __future__ import annotations

__all__ = ["DatasetDataSource"]

import logging
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from coola.utils import str_indent, str_mapping
from torch.utils.data import Dataset

from gravitorch.creators.dataloader.base import BaseDataLoaderCreator
from gravitorch.creators.dataloader.factory import setup_dataloader_creator
from gravitorch.datasets import setup_dataset
from gravitorch.datasources.base import BaseDataSource, IterableNotFoundError
from gravitorch.utils.asset import AssetManager

if TYPE_CHECKING:
    from gravitorch.engines import BaseEngine

logger = logging.getLogger(__name__)

T = TypeVar("T")


class DatasetDataSource(BaseDataSource):
    r"""Implements a datasource that uses regular PyTorch datasets and
    data loaders.

    To create a data loader, the user should indicate the dataset and
    the data loader creator. Note that the regular PyTorch data
    loader needs a ``Dataset`` object has input.

    Args:
    ----
        datasets (dict): Specifies the datasets to initialize. Each
            key indicates the dataset name. It is possible to give a
            ``Dataset`` object, or the configuration of a ``Dataset``
            object.
        dataloader_creators (dict): Specifies the data loader
            creators to initialize. Each key indicates a data loader
            creator name. For example if you want to create a data
            loader for ``'train'`` ID, the dictionary has to have a
            key ``'train'``. The value can be a
            ``BaseDataLoaderCreator`` object, or its configuration,
            or ``None``. ``None`` means a default data loader will be
            created. Each data loader creator takes a ``Dataset``
            object as input, so you need to specify a dataset with the
            same name.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.datasources import DatasetDataSource
        >>> datasource = DatasetDataSource(
        ...     datasets={
        ...         "train": {
        ...             "_target_": "gravitorch.datasets.ExampleDataset",
        ...             "examples": [1, 2, 3, 4],
        ...         },
        ...         "eval": {
        ...             "_target_": "gravitorch.datasets.ExampleDataset",
        ...             "examples": [5, 6, 7],
        ...         },
        ...     },
        ...     dataloader_creators={},
        ... )
        >>> datasource
        DatasetDataSource(
          (datasets):
            (train): ExampleDataset(num_examples=4)
            (eval): ExampleDataset(num_examples=3)
          (dataloader_creators):
        )
    """

    def __init__(
        self,
        datasets: Mapping[str, Dataset | dict],
        dataloader_creators: dict[str, BaseDataLoaderCreator | dict | None],
    ) -> None:
        self._asset_manager = AssetManager()

        logger.info("Initializing the datasets...")
        self._datasets = {key: setup_dataset(dataset) for key, dataset in datasets.items()}
        logger.info(f"datasets:\n{str_mapping(self._datasets)}")
        for name, dataset in self._datasets.items():
            self._asset_manager.add_asset(f"{name}_dataset", dataset)

        logger.info("Initializing the data loader creators...")
        self._dataloader_creators = {
            key: setup_dataloader_creator(creator) for key, creator in dataloader_creators.items()
        }
        logger.info(f"data loader creators:\n{str_mapping(self._dataloader_creators)}")
        self._check()

    def __repr__(self) -> str:
        args = str_indent(
            str_mapping(
                {
                    "datasets": "\n" + str_mapping(self._datasets) if self._datasets else "",
                    "dataloader_creators": "\n" + str_mapping(self._dataloader_creators)
                    if self._dataloader_creators
                    else "",
                }
            )
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def attach(self, engine: BaseEngine) -> None:
        logger.info("Attach the datasource to an engine")

    def get_asset(self, asset_id: str) -> Any:
        return self._asset_manager.get_asset(asset_id)

    def has_asset(self, asset_id: str) -> bool:
        return self._asset_manager.has_asset(asset_id)

    def get_iterable(self, iter_id: str, engine: BaseEngine | None = None) -> Iterable[T]:
        if not self.has_iterable(iter_id):
            raise IterableNotFoundError(f"{iter_id} does not exist")
        return self._dataloader_creators[iter_id].create(
            dataset=self._datasets[iter_id], engine=engine
        )

    def has_iterable(self, iter_id: str) -> bool:
        return iter_id in self._dataloader_creators

    def _check(self) -> None:
        # Verify each data loader creator has a dataset
        for key in self._dataloader_creators:
            if key not in self._datasets:
                logger.warning(f"Missing '{key}' dataset for its associated data loader creator")
        # Verify each dataset has a data loader creator
        for key in self._datasets:
            if key not in self._dataloader_creators:
                logger.warning(f"Missing '{key}' data loader creator for its associated dataset")
