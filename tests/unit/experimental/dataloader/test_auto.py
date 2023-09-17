from __future__ import annotations

from unittest.mock import patch

import torch
from pytest import fixture, mark
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from gravitorch.creators.dataloader import (
    AutoDataLoaderCreator,
    DistributedDataLoaderCreator,
    VanillaDataLoaderCreator,
)
from gravitorch.creators.dataset import DatasetCreator


class FakeDataset(Dataset):
    def __len__(self) -> int:
        return 20

    def __getitem__(self, item: int) -> Tensor:
        return torch.ones(5).mul(item)


@fixture
def dataset() -> Dataset:
    return FakeDataset()


###########################################
#     Tests for AutoDataLoaderCreator     #
###########################################


def test_auto_dataloader_creator_str(dataset: Dataset) -> None:
    assert str(AutoDataLoaderCreator(dataset)).startswith("AutoDataLoaderCreator(")


def test_auto_dataloader_creator_dataset(dataset: Dataset) -> None:
    dataloader = AutoDataLoaderCreator(dataset, batch_size=8).create()
    assert isinstance(dataloader, DataLoader)
    assert dataloader.dataset is dataset
    batch = next(iter(dataloader))
    assert batch.equal(
        torch.tensor(
            [
                [0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1],
                [2, 2, 2, 2, 2],
                [3, 3, 3, 3, 3],
                [4, 4, 4, 4, 4],
                [5, 5, 5, 5, 5],
                [6, 6, 6, 6, 6],
                [7, 7, 7, 7, 7],
            ],
            dtype=torch.float,
        )
    )


def test_auto_dataloader_creator_dataset_creator(dataset: Dataset) -> None:
    dataloader = AutoDataLoaderCreator(DatasetCreator(dataset), batch_size=8).create()
    assert isinstance(dataloader, DataLoader)
    assert dataloader.dataset is dataset
    batch = next(iter(dataloader))
    assert batch.equal(
        torch.tensor(
            [
                [0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1],
                [2, 2, 2, 2, 2],
                [3, 3, 3, 3, 3],
                [4, 4, 4, 4, 4],
                [5, 5, 5, 5, 5],
                [6, 6, 6, 6, 6],
                [7, 7, 7, 7, 7],
            ],
            dtype=torch.float,
        )
    )


@mark.parametrize("batch_size", (1, 2, 4))
def test_auto_dataloader_creator_batch_size(dataset: Dataset, batch_size: int) -> None:
    dataloader = AutoDataLoaderCreator(dataset, batch_size=batch_size).create()
    assert isinstance(dataloader, DataLoader)
    assert dataloader.batch_size == batch_size
    batch = next(iter(dataloader))
    assert torch.is_tensor(batch)
    assert batch.shape == (batch_size, 5)


@mark.parametrize("batch_size", (1, 2, 4))
@patch("gravitorch.creators.dataloader.distributed.dist.is_distributed", lambda *args: False)
def test_auto_dataloader_creator_non_distributed(dataset: Dataset, batch_size: int) -> None:
    creator = AutoDataLoaderCreator(dataset, batch_size=batch_size)
    assert isinstance(creator._dataloader, VanillaDataLoaderCreator)
    dataloader = creator.create()
    assert isinstance(dataloader, DataLoader)
    assert dataloader.batch_size == batch_size


@mark.parametrize("batch_size", (1, 2, 4))
@patch("gravitorch.creators.dataloader.distributed.dist.is_distributed", lambda *args: True)
def test_auto_dataloader_creator_distributed(dataset: Dataset, batch_size: int) -> None:
    creator = AutoDataLoaderCreator(dataset, batch_size=batch_size)
    assert isinstance(creator._dataloader, DistributedDataLoaderCreator)
    dataloader = creator.create()
    assert isinstance(dataloader, DataLoader)
    assert dataloader.batch_size == batch_size
