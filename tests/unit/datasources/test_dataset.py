import logging
from unittest.mock import Mock

import torch
from coola import objects_are_equal
from objectory import OBJECT_TARGET
from pytest import LogCaptureFixture, raises
from torch.utils.data import DataLoader, Dataset

from gravitorch.creators.dataloader import AutoDataLoaderCreator, BaseDataLoaderCreator
from gravitorch.data.datasets import ExampleDataset
from gravitorch.datasources import DatasetDataSource, LoaderNotFoundError
from gravitorch.engines import BaseEngine
from gravitorch.utils.asset import AssetNotFoundError

#######################################
#     Tests for DatasetDataSource     #
#######################################


def test_dataset_datasource_str() -> None:
    assert str(DatasetDataSource(datasets={}, data_loader_creators={})).startswith(
        "DatasetDataSource("
    )


def test_dataset_datasource_datasets() -> None:
    datasource = DatasetDataSource(
        datasets={
            "train": {
                OBJECT_TARGET: "gravitorch.data.datasets.ExampleDataset",
                "examples": [1, 2, 3, 4],
            },
            "val": ExampleDataset(["a", "b", "c"]),
        },
        data_loader_creators={},
    )
    assert len(datasource._datasets) == 2
    assert isinstance(datasource._datasets["train"], ExampleDataset)
    assert list(datasource._datasets["train"]) == [1, 2, 3, 4]
    assert isinstance(datasource._datasets["val"], ExampleDataset)
    assert list(datasource._datasets["val"]) == ["a", "b", "c"]


def test_dataset_datasource_data_loader_creators() -> None:
    datasource = DatasetDataSource(
        datasets={},
        data_loader_creators={
            "train": AutoDataLoaderCreator(),
            "val": {OBJECT_TARGET: "gravitorch.creators.dataloader.AutoDataLoaderCreator"},
            "test": None,
        },
    )
    assert len(datasource._data_loader_creators) == 3
    assert isinstance(datasource._data_loader_creators["train"], AutoDataLoaderCreator)
    assert isinstance(datasource._data_loader_creators["val"], AutoDataLoaderCreator)
    assert isinstance(datasource._data_loader_creators["test"], AutoDataLoaderCreator)


def test_dataset_datasource_attach(caplog: LogCaptureFixture) -> None:
    datasource = DatasetDataSource(datasets={}, data_loader_creators={})
    with caplog.at_level(logging.INFO):
        datasource.attach(engine=Mock(spec=BaseEngine))
        assert len(caplog.messages) >= 1


def test_dataset_datasource_get_asset_exists() -> None:
    datasource = DatasetDataSource(datasets={}, data_loader_creators={})
    datasource._asset_manager.add_asset("something", 2)
    assert datasource.get_asset("something") == 2


def test_dataset_datasource_get_asset_does_not_exist() -> None:
    datasource = DatasetDataSource(datasets={}, data_loader_creators={})
    with raises(AssetNotFoundError, match="The asset 'something' does not exist"):
        datasource.get_asset("something")


def test_dataset_datasource_has_asset_true() -> None:
    datasource = DatasetDataSource(datasets={}, data_loader_creators={})
    datasource._asset_manager.add_asset("something", 1)
    assert datasource.has_asset("something")


def test_dataset_datasource_has_asset_false() -> None:
    datasource = DatasetDataSource(datasets={}, data_loader_creators={})
    assert not datasource.has_asset("something")


def test_dataset_datasource_get_data_loader_train() -> None:
    datasource = DatasetDataSource(
        datasets={
            "train": ExampleDataset([1, 2, 3, 4]),
            "val": ExampleDataset(["a", "b", "c"]),
        },
        data_loader_creators={
            "train": AutoDataLoaderCreator(batch_size=4, shuffle=False),
            "val": AutoDataLoaderCreator(batch_size=2, shuffle=False),
        },
    )
    loader = datasource.get_data_loader("train")
    assert isinstance(loader, DataLoader)
    assert objects_are_equal(tuple(loader), (torch.tensor([1, 2, 3, 4]),))


def test_dataset_datasource_get_data_loader_val() -> None:
    datasource = DatasetDataSource(
        datasets={
            "train": ExampleDataset([1, 2, 3, 4]),
            "val": ExampleDataset(["a", "b", "c"]),
        },
        data_loader_creators={
            "train": AutoDataLoaderCreator(batch_size=4, shuffle=False),
            "val": AutoDataLoaderCreator(batch_size=2, shuffle=False),
        },
    )
    loader = datasource.get_data_loader("val")
    assert isinstance(loader, DataLoader)
    assert tuple(loader) == (["a", "b"], ["c"])


def test_dataset_datasource_get_data_loader_missing() -> None:
    datasource = DatasetDataSource(datasets={}, data_loader_creators={})
    with raises(LoaderNotFoundError):
        datasource.get_data_loader("missing")


def test_dataset_datasource_get_data_loader_with_engine() -> None:
    engine = Mock(spec=BaseEngine)
    dataset = Mock(spec=Dataset)
    data_loader_creator = Mock(spec=BaseDataLoaderCreator)
    datasource = DatasetDataSource(
        datasets={"train": dataset},
        data_loader_creators={"train": data_loader_creator},
    )
    datasource.get_data_loader("train", engine)
    data_loader_creator.create.assert_called_once_with(dataset=dataset, engine=engine)


def test_dataset_datasource_get_data_loader_without_engine() -> None:
    dataset = Mock(spec=Dataset)
    data_loader_creator = Mock(spec=BaseDataLoaderCreator)
    datasource = DatasetDataSource(
        datasets={"train": dataset},
        data_loader_creators={"train": data_loader_creator},
    )
    datasource.get_data_loader("train")
    data_loader_creator.create.assert_called_once_with(dataset=dataset, engine=None)


def test_dataset_datasource_has_data_loader_true() -> None:
    datasource = DatasetDataSource(
        datasets={},
        data_loader_creators={"train": AutoDataLoaderCreator(batch_size=4)},
    )
    assert datasource.has_data_loader("train")


def test_dataset_datasource_has_data_loader_false() -> None:
    datasource = DatasetDataSource(datasets={}, data_loader_creators={})
    assert not datasource.has_data_loader("missing")


def test_dataset_datasource_load_state_dict() -> None:
    datasource = DatasetDataSource(datasets={}, data_loader_creators={})
    datasource.load_state_dict({})


def test_dataset_datasource_state_dict() -> None:
    datasource = DatasetDataSource(datasets={}, data_loader_creators={})
    assert datasource.state_dict() == {}


def test_dataset_datasource_check_missing_dataset(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(level=logging.WARNING):
        DatasetDataSource(
            datasets={}, data_loader_creators={"train": Mock(spec=BaseDataLoaderCreator)}
        )
        assert len(caplog.messages) == 1


def test_dataset_datasource_check_missing_data_loader_creator(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(level=logging.WARNING):
        DatasetDataSource(datasets={"train": Mock(spec=Dataset)}, data_loader_creators={})
        assert len(caplog.messages) == 1
