from __future__ import annotations

from objectory import OBJECT_TARGET
from pytest import fixture, mark
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

from gravitorch.dataloaders import (
    create_dataloader,
    is_dataloader_config,
    setup_dataloader,
)
from gravitorch.datasets import DummyMultiClassDataset, ExampleDataset
from gravitorch.testing import torchdata_available

#######################################
#     Tests for create_dataloader     #
#######################################


@fixture(scope="module")
def dataset() -> Dataset:
    return DummyMultiClassDataset(num_examples=10, num_classes=2, feature_size=4)


@mark.parametrize(
    "dataset",
    (
        DummyMultiClassDataset(num_examples=10, num_classes=2, feature_size=4),
        {
            OBJECT_TARGET: "gravitorch.datasets.DummyMultiClassDataset",
            "num_examples": 10,
            "num_classes": 2,
            "feature_size": 4,
        },
    ),
)
def test_create_dataloader_dataset(dataset: Dataset | dict) -> None:
    dataloader = create_dataloader(dataset)
    assert isinstance(dataloader, DataLoader)
    assert isinstance(dataloader.dataset, DummyMultiClassDataset)
    assert dataloader.batch_size == 1
    assert isinstance(dataloader.sampler, SequentialSampler)


@mark.parametrize("batch_size", (1, 2))
def test_create_dataloader_batch_size(dataset: Dataset, batch_size: int) -> None:
    dataloader = create_dataloader(dataset, batch_size=batch_size)
    assert isinstance(dataloader, DataLoader)
    assert isinstance(dataloader.dataset, DummyMultiClassDataset)
    assert dataloader.batch_size == batch_size


def test_create_dataloader_sampler(dataset: Dataset) -> None:
    dataloader = create_dataloader(dataset, sampler=RandomSampler(dataset))
    assert isinstance(dataloader, DataLoader)
    assert isinstance(dataloader.dataset, DummyMultiClassDataset)
    assert dataloader.batch_size == 1
    assert isinstance(dataloader.sampler, RandomSampler)


def test_create_dataloader_config(dataset: Dataset) -> None:
    dataloader = create_dataloader(
        dataset,
        sampler={OBJECT_TARGET: "torch.utils.data.RandomSampler", "data_source": dataset},
    )
    assert isinstance(dataloader, DataLoader)
    assert isinstance(dataloader.dataset, DummyMultiClassDataset)
    assert dataloader.batch_size == 1
    assert isinstance(dataloader.sampler, RandomSampler)


##########################################
#     Tests for is_dataloader_config     #
##########################################


def test_is_dataloader_config_true() -> None:
    assert is_dataloader_config({OBJECT_TARGET: "torch.utils.data.DataLoader"})


def test_is_dataloader_config_false() -> None:
    assert not is_dataloader_config({OBJECT_TARGET: "torch.nn.Identity"})


@torchdata_available
def test_is_dataloader_config_false_dataloader2() -> None:
    assert not is_dataloader_config({OBJECT_TARGET: "torchdata.dataloader2.DataLoader2"})


######################################
#     Tests for setup_dataloader     #
######################################


def test_setup_dataloader_object() -> None:
    creator = DataLoader(ExampleDataset((1, 2, 3, 4, 5)))
    assert setup_dataloader(creator) is creator


def test_setup_dataloader_dict() -> None:
    assert isinstance(
        setup_dataloader(
            {
                OBJECT_TARGET: "torch.utils.data.DataLoader",
                "dataset": {
                    OBJECT_TARGET: "gravitorch.datasets.ExampleDataset",
                    "examples": (1, 2, 3, 4, 5),
                },
            },
        ),
        DataLoader,
    )
