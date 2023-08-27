from __future__ import annotations

__all__ = ["DatasetDataSource"]

import logging
from collections.abc import Iterable, Mapping
from typing import Any, TypeVar

from coola.utils import str_indent, str_mapping
from torch.utils.data import Dataset

from gravitorch.creators.dataloader.base import BaseDataLoaderCreator
from gravitorch.creators.dataloader.factory import setup_dataloader_creator
from gravitorch.data.datasets.factory import setup_dataset
from gravitorch.datasources.base import BaseDataSource, LoaderNotFoundError
from gravitorch.engines.base import BaseEngine
from gravitorch.utils.asset import AssetManager

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

        logger.info("Initializing the data loader creators...")
        self._dataloader_creators = {
            key: setup_dataloader_creator(creator) for key, creator in dataloader_creators.items()
        }
        logger.info(f"data loader creators:\n{str_mapping(self._dataloader_creators)}")
        self._check()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(\n"
            "  datasets:\n"
            f"    {str_indent(str_mapping(self._datasets), num_spaces=4)}\n"
            "  dataloader_creators:\n"
            f"    {str_indent(str_mapping(self._dataloader_creators), num_spaces=4)}"
            "\n)"
        )

    def attach(self, engine: BaseEngine) -> None:
        logger.info("Attach the datasource to an engine")

    def get_asset(self, asset_id: str) -> Any:
        return self._asset_manager.get_asset(asset_id)

    def has_asset(self, asset_id: str) -> bool:
        return self._asset_manager.has_asset(asset_id)

    def get_dataloader(self, loader_id: str, engine: BaseEngine | None = None) -> Iterable[T]:
        if not self.has_dataloader(loader_id):
            raise LoaderNotFoundError(f"{loader_id} does not exist")
        return self._dataloader_creators[loader_id].create(
            dataset=self._datasets[loader_id], engine=engine
        )

    def has_dataloader(self, loader_id: str) -> bool:
        return loader_id in self._dataloader_creators

    def _check(self) -> None:
        # Verify each data loader creator has a dataset
        for key in self._dataloader_creators:
            if key not in self._datasets:
                logger.warning(f"Missing '{key}' dataset for its associated data loader creator")
        # Verify each dataset has a data loader creator
        for key in self._datasets:
            if key not in self._dataloader_creators:
                logger.warning(f"Missing '{key}' data loader creator for its associated dataset")
