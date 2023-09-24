from __future__ import annotations

__all__ = ["DataLoaderCreator", "VanillaDataLoaderCreator"]

from typing import TYPE_CHECKING, TypeVar

from torch.utils.data import DataLoader, Dataset

from gravitorch.creators.dataset import (
    BaseDatasetCreator,
    DatasetCreator,
    setup_dataset_creator,
)
from gravitorch.dataloaders.factory import create_dataloader, setup_dataloader
from gravitorch.datasets.factory import is_dataset_config
from gravitorch.experimental.dataloader.base import BaseDataLoaderCreator
from gravitorch.utils.format import str_indent, str_mapping
from gravitorch.utils.seed import get_torch_generator

if TYPE_CHECKING:
    from gravitorch.engines import BaseEngine

T = TypeVar("T")


class DataLoaderCreator(BaseDataLoaderCreator[T]):
    r"""Implements a simple dataloader creator.

    Args:
    ----
        dataloader (``torch.utils.data.DataLoader`` or dict): Specifies
            the dataloader or its configuration.
        cache (bool, optional): If ``True``, the dataloader is created
            only the first time, and then the same data is returned
            for each call to the ``create`` method.
            Default: ``False``

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.experimental.dataloader import DataLoaderCreator
        >>> from gravitorch.datasets import ExampleDataset
        >>> creator = DataLoaderCreator(
        ...     {
        ...         "_target_": "torch.utils.data.DataLoader",
        ...         "dataset": ExampleDataset((1, 2, 3, 4)),
        ...     },
        ... )
        >>> creator.create()
        <torch.utils.data.dataloader.DataLoader object at 0x...>
    """

    def __init__(self, dataloader: DataLoader | dict, cache: bool = False) -> None:
        self._dataloader = dataloader
        self._cache = bool(cache)

    def __repr__(self) -> str:
        config = {"dataloader": self._dataloader, "cache": self._cache}
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  {str_indent(str_mapping(config, sorted_keys=True))}"
            "\n)"
        )

    def create(self, engine: BaseEngine | None = None) -> DataLoader[T]:
        dataloader = setup_dataloader(self._dataloader)
        if self._cache:
            self._dataloader = dataloader
        return dataloader


class VanillaDataLoaderCreator(BaseDataLoaderCreator[T]):
    r"""Implements a simple dataloader creator.

    Args:
    ----
        dataset (``torch.utils.data.Dataset`` or ``BaseDatasetCreator``
            or ``dict``): Specifies a dataset (or its configuration)
            or a dataset creator (or its configuration).
        seed (int, optional): Specifies the random seed used to
            reproduce the shuffling of the samples. Default: ``0``
        **kwargs: See ``torch.utils.data.DataLoader`` documentation.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.experimental.dataloader import VanillaDataLoaderCreator
        >>> creator = VanillaDataLoaderCreator(
        ...     {
        ...         "_target_": "gravitorch.datasets.DummyMultiClassDataset",
        ...         "num_examples": 10,
        ...         "num_classes": 2,
        ...         "feature_size": 4,
        ...     }
        ... )
        >>> creator.create()
        <torch.utils.data.dataloader.DataLoader object at 0x...>
    """

    def __init__(
        self, dataset: Dataset | BaseDatasetCreator | dict, seed: int = 0, **kwargs
    ) -> None:
        if isinstance(dataset, Dataset) or (
            isinstance(dataset, dict) and is_dataset_config(dataset)
        ):
            dataset = DatasetCreator(dataset)
        self._dataset = setup_dataset_creator(dataset)
        self._seed = int(seed)
        self._kwargs = kwargs

    def __repr__(self) -> str:
        config = {"dataset": self._dataset, "seed": self._seed} | self._kwargs
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  {str_indent(str_mapping(config, sorted_keys=True))}\n)"
        )

    def create(self, engine: BaseEngine | None = None) -> DataLoader[T]:
        epoch = 0 if engine is None else engine.epoch
        return create_dataloader(
            self._dataset.create(engine),
            generator=get_torch_generator(self._seed + epoch),
            **self._kwargs,
        )
