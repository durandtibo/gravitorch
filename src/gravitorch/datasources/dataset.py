__all__ = ["DatasetDataSource"]

import logging
from collections.abc import Iterable
from typing import Any, Optional, TypeVar, Union

from torch.utils.data import Dataset

from gravitorch.creators.dataloader.base import BaseDataLoaderCreator
from gravitorch.creators.dataloader.utils import setup_data_loader_creator
from gravitorch.data.datasets import setup_dataset
from gravitorch.datasources.base import BaseDataSource, LoaderNotFoundError
from gravitorch.engines.base import BaseEngine
from gravitorch.utils.asset import AssetManager
from gravitorch.utils.format import str_indent, to_torch_mapping_str

logger = logging.getLogger(__name__)

T = TypeVar("T")


class DatasetDataSource(BaseDataSource):
    r"""Implements a data source that uses regular PyTorch datasets and data
    loaders.

    To create a data loader, the user should indicate the dataset and
    the data loader creator. Note that the regular PyTorch data
    loader needs a ``Dataset`` object has input.

    Args:
        datasets (dict): Specifies the datasets to initialize. Each
            key indicates the dataset name. It is possible to give a
            ``Dataset`` object, or the configuration of a ``Dataset``
            object.
        data_loader_creators (dict): Specifies the data loader
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
        datasets: dict[str, Union[Dataset, dict]],
        data_loader_creators: dict[str, Union[BaseDataLoaderCreator, dict, None]],
    ):
        self._asset_manager = AssetManager()

        logger.info("Initializing the datasets...")
        self._datasets = {key: setup_dataset(dataset) for key, dataset in datasets.items()}
        logger.info(f"datasets:\n{to_torch_mapping_str(self._datasets)}")

        logger.info("Initializing the data loader creators...")
        self._data_loader_creators = {
            key: setup_data_loader_creator(creator) for key, creator in data_loader_creators.items()
        }
        logger.info(f"data loader creators:\n{to_torch_mapping_str(self._data_loader_creators)}")
        self._check()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(\n"
            "  datasets:\n"
            f"    {str_indent(to_torch_mapping_str(self._datasets), num_spaces=4)}\n"
            "  data_loader_creators:\n"
            f"    {str_indent(to_torch_mapping_str(self._data_loader_creators), num_spaces=4)}"
            "\n)"
        )

    def attach(self, engine: BaseEngine) -> None:
        logger.info("Attach the data source to an engine")

    def get_asset(self, asset_id: str) -> Any:
        return self._asset_manager.get_asset(asset_id)

    def has_asset(self, asset_id: str) -> bool:
        return self._asset_manager.has_asset(asset_id)

    def get_data_loader(self, loader_id: str, engine: Optional[BaseEngine] = None) -> Iterable[T]:
        if not self.has_data_loader(loader_id):
            raise LoaderNotFoundError(f"{loader_id} does not exist")
        return self._data_loader_creators[loader_id].create(
            dataset=self._datasets[loader_id], engine=engine
        )

    def has_data_loader(self, loader_id: str) -> bool:
        return loader_id in self._data_loader_creators

    def _check(self) -> None:
        # Verify each data loader creator has a dataset
        for key in self._data_loader_creators:
            if key not in self._datasets:
                logger.warning(f"Missing '{key}' dataset for its associated data loader creator")
        # Verify each dataset has a data loader creator
        for key in self._datasets:
            if key not in self._data_loader_creators:
                logger.warning(f"Missing '{key}' data loader creator for its associated dataset")
