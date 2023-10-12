import logging
from unittest.mock import Mock

import torch
from coola import objects_are_equal
from objectory import OBJECT_TARGET
from pytest import LogCaptureFixture, raises
from torch.utils.data import DataLoader, Dataset

from gravitorch.creators.dataloader import AutoDataLoaderCreator, BaseDataLoaderCreator
from gravitorch.datasets import ExampleDataset
from gravitorch.datasources import DatasetDataSource, DataStreamNotFoundError
from gravitorch.engines import BaseEngine
from gravitorch.utils.asset import AssetNotFoundError

#######################################
#     Tests for DatasetDataSource     #
#######################################


def test_dataset_datasource_str() -> None:
    assert str(DatasetDataSource(datasets={}, dataloader_creators={})).startswith(
        "DatasetDataSource("
    )


def test_dataset_datasource_datasets() -> None:
    datasource = DatasetDataSource(
        datasets={
            "train": {
                OBJECT_TARGET: "gravitorch.datasets.ExampleDataset",
                "examples": [1, 2, 3, 4],
            },
            "val": ExampleDataset(["a", "b", "c"]),
        },
        dataloader_creators={},
    )
    assert len(datasource._datasets) == 2
    assert isinstance(datasource._datasets["train"], ExampleDataset)
    assert list(datasource._datasets["train"]) == [1, 2, 3, 4]
    assert isinstance(datasource._datasets["val"], ExampleDataset)
    assert list(datasource._datasets["val"]) == ["a", "b", "c"]


def test_dataset_datasource_dataloader_creators() -> None:
    datasource = DatasetDataSource(
        datasets={},
        dataloader_creators={
            "train": AutoDataLoaderCreator(),
            "val": {OBJECT_TARGET: "gravitorch.creators.dataloader.AutoDataLoaderCreator"},
            "test": None,
        },
    )
    assert len(datasource._dataloader_creators) == 3
    assert isinstance(datasource._dataloader_creators["train"], AutoDataLoaderCreator)
    assert isinstance(datasource._dataloader_creators["val"], AutoDataLoaderCreator)
    assert isinstance(datasource._dataloader_creators["test"], AutoDataLoaderCreator)


def test_dataset_datasource_attach(caplog: LogCaptureFixture) -> None:
    datasource = DatasetDataSource(datasets={}, dataloader_creators={})
    with caplog.at_level(logging.INFO):
        datasource.attach(engine=Mock(spec=BaseEngine))
        assert len(caplog.messages) >= 1


def test_dataset_datasource_get_asset_exists() -> None:
    datasource = DatasetDataSource(
        datasets={"train": ExampleDataset(["a", "b", "c"])}, dataloader_creators={}
    )
    assert datasource.get_asset("train_dataset").equal(ExampleDataset(["a", "b", "c"]))


def test_dataset_datasource_get_asset_does_not_exist() -> None:
    datasource = DatasetDataSource(datasets={}, dataloader_creators={})
    with raises(AssetNotFoundError, match="The asset 'missing' does not exist"):
        datasource.get_asset("missing")


def test_dataset_datasource_has_asset_true() -> None:
    datasource = DatasetDataSource(
        datasets={"train": ExampleDataset(["a", "b", "c"])}, dataloader_creators={}
    )
    assert datasource.has_asset("train_dataset")


def test_dataset_datasource_has_asset_false() -> None:
    datasource = DatasetDataSource(datasets={}, dataloader_creators={})
    assert not datasource.has_asset("missing")


def test_dataset_datasource_get_datastream_train() -> None:
    datasource = DatasetDataSource(
        datasets={
            "train": ExampleDataset([1, 2, 3, 4]),
            "val": ExampleDataset(["a", "b", "c"]),
        },
        dataloader_creators={
            "train": AutoDataLoaderCreator(batch_size=4, shuffle=False),
            "val": AutoDataLoaderCreator(batch_size=2, shuffle=False),
        },
    )
    loader = datasource.get_datastream("train")
    assert isinstance(loader, DataLoader)
    assert objects_are_equal(tuple(loader), (torch.tensor([1, 2, 3, 4]),))


def test_dataset_datasource_get_datastream_val() -> None:
    datasource = DatasetDataSource(
        datasets={
            "train": ExampleDataset([1, 2, 3, 4]),
            "val": ExampleDataset(["a", "b", "c"]),
        },
        dataloader_creators={
            "train": AutoDataLoaderCreator(batch_size=4, shuffle=False),
            "val": AutoDataLoaderCreator(batch_size=2, shuffle=False),
        },
    )
    loader = datasource.get_datastream("val")
    assert isinstance(loader, DataLoader)
    assert tuple(loader) == (["a", "b"], ["c"])


def test_dataset_datasource_get_datastream_missing() -> None:
    datasource = DatasetDataSource(datasets={}, dataloader_creators={})
    with raises(DataStreamNotFoundError):
        datasource.get_datastream("missing")


def test_dataset_datasource_get_datastream_with_engine() -> None:
    engine = Mock(spec=BaseEngine)
    dataset = Mock(spec=Dataset)
    dataloader_creator = Mock(spec=BaseDataLoaderCreator)
    datasource = DatasetDataSource(
        datasets={"train": dataset},
        dataloader_creators={"train": dataloader_creator},
    )
    datasource.get_datastream("train", engine)
    dataloader_creator.create.assert_called_once_with(dataset=dataset, engine=engine)


def test_dataset_datasource_get_datastream_without_engine() -> None:
    dataset = Mock(spec=Dataset)
    dataloader_creator = Mock(spec=BaseDataLoaderCreator)
    datasource = DatasetDataSource(
        datasets={"train": dataset},
        dataloader_creators={"train": dataloader_creator},
    )
    datasource.get_datastream("train")
    dataloader_creator.create.assert_called_once_with(dataset=dataset, engine=None)


def test_dataset_datasource_has_datastream_true() -> None:
    datasource = DatasetDataSource(
        datasets={},
        dataloader_creators={"train": AutoDataLoaderCreator(batch_size=4)},
    )
    assert datasource.has_datastream("train")


def test_dataset_datasource_has_datastream_false() -> None:
    datasource = DatasetDataSource(datasets={}, dataloader_creators={})
    assert not datasource.has_datastream("missing")


def test_dataset_datasource_load_state_dict() -> None:
    datasource = DatasetDataSource(datasets={}, dataloader_creators={})
    datasource.load_state_dict({})


def test_dataset_datasource_state_dict() -> None:
    datasource = DatasetDataSource(datasets={}, dataloader_creators={})
    assert datasource.state_dict() == {}


def test_dataset_datasource_check_missing_dataset(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(level=logging.WARNING):
        DatasetDataSource(
            datasets={}, dataloader_creators={"train": Mock(spec=BaseDataLoaderCreator)}
        )
        assert len(caplog.messages) == 1


def test_dataset_datasource_check_missing_dataloader_creator(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(level=logging.WARNING):
        DatasetDataSource(datasets={"train": Mock(spec=Dataset)}, dataloader_creators={})
        assert len(caplog.messages) == 1
