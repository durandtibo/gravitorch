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


def test_dataset_data_source_str() -> None:
    assert str(DatasetDataSource(datasets={}, data_loader_creators={})).startswith(
        "DatasetDataSource("
    )


def test_dataset_data_source_datasets() -> None:
    data_source = DatasetDataSource(
        datasets={
            "train": {
                OBJECT_TARGET: "gravitorch.data.datasets.ExampleDataset",
                "examples": [1, 2, 3, 4],
            },
            "val": ExampleDataset(["a", "b", "c"]),
        },
        data_loader_creators={},
    )
    assert len(data_source._datasets) == 2
    assert isinstance(data_source._datasets["train"], ExampleDataset)
    assert list(data_source._datasets["train"]) == [1, 2, 3, 4]
    assert isinstance(data_source._datasets["val"], ExampleDataset)
    assert list(data_source._datasets["val"]) == ["a", "b", "c"]


def test_dataset_data_source_data_loader_creators() -> None:
    data_source = DatasetDataSource(
        datasets={},
        data_loader_creators={
            "train": AutoDataLoaderCreator(),
            "val": {OBJECT_TARGET: "gravitorch.creators.dataloader.AutoDataLoaderCreator"},
            "test": None,
        },
    )
    assert len(data_source._data_loader_creators) == 3
    assert isinstance(data_source._data_loader_creators["train"], AutoDataLoaderCreator)
    assert isinstance(data_source._data_loader_creators["val"], AutoDataLoaderCreator)
    assert isinstance(data_source._data_loader_creators["test"], AutoDataLoaderCreator)


def test_dataset_data_source_attach(caplog: LogCaptureFixture) -> None:
    data_source = DatasetDataSource(datasets={}, data_loader_creators={})
    with caplog.at_level(logging.INFO):
        data_source.attach(engine=Mock(spec=BaseEngine))
        assert len(caplog.messages) >= 1


def test_dataset_data_source_get_asset_exists() -> None:
    data_source = DatasetDataSource(datasets={}, data_loader_creators={})
    data_source._asset_manager.add_asset("something", 2)
    assert data_source.get_asset("something") == 2


def test_dataset_data_source_get_asset_does_not_exist() -> None:
    data_source = DatasetDataSource(datasets={}, data_loader_creators={})
    with raises(AssetNotFoundError, match="The asset 'something' does not exist"):
        data_source.get_asset("something")


def test_dataset_data_source_has_asset_true() -> None:
    data_source = DatasetDataSource(datasets={}, data_loader_creators={})
    data_source._asset_manager.add_asset("something", 1)
    assert data_source.has_asset("something")


def test_dataset_data_source_has_asset_false() -> None:
    data_source = DatasetDataSource(datasets={}, data_loader_creators={})
    assert not data_source.has_asset("something")


def test_dataset_data_source_get_data_loader_train() -> None:
    data_source = DatasetDataSource(
        datasets={
            "train": ExampleDataset([1, 2, 3, 4]),
            "val": ExampleDataset(["a", "b", "c"]),
        },
        data_loader_creators={
            "train": AutoDataLoaderCreator(batch_size=4, shuffle=False),
            "val": AutoDataLoaderCreator(batch_size=2, shuffle=False),
        },
    )
    loader = data_source.get_data_loader("train")
    assert isinstance(loader, DataLoader)
    assert objects_are_equal(tuple(loader), (torch.tensor([1, 2, 3, 4]),))


def test_dataset_data_source_get_data_loader_val() -> None:
    data_source = DatasetDataSource(
        datasets={
            "train": ExampleDataset([1, 2, 3, 4]),
            "val": ExampleDataset(["a", "b", "c"]),
        },
        data_loader_creators={
            "train": AutoDataLoaderCreator(batch_size=4, shuffle=False),
            "val": AutoDataLoaderCreator(batch_size=2, shuffle=False),
        },
    )
    loader = data_source.get_data_loader("val")
    assert isinstance(loader, DataLoader)
    assert tuple(loader) == (["a", "b"], ["c"])


def test_dataset_data_source_get_data_loader_missing() -> None:
    data_source = DatasetDataSource(datasets={}, data_loader_creators={})
    with raises(LoaderNotFoundError):
        data_source.get_data_loader("missing")


def test_dataset_data_source_get_data_loader_with_engine() -> None:
    engine = Mock(spec=BaseEngine)
    dataset = Mock(spec=Dataset)
    data_loader_creator = Mock(spec=BaseDataLoaderCreator)
    data_source = DatasetDataSource(
        datasets={"train": dataset},
        data_loader_creators={"train": data_loader_creator},
    )
    data_source.get_data_loader("train", engine)
    data_loader_creator.create.assert_called_once_with(dataset=dataset, engine=engine)


def test_dataset_data_source_get_data_loader_without_engine() -> None:
    dataset = Mock(spec=Dataset)
    data_loader_creator = Mock(spec=BaseDataLoaderCreator)
    data_source = DatasetDataSource(
        datasets={"train": dataset},
        data_loader_creators={"train": data_loader_creator},
    )
    data_source.get_data_loader("train")
    data_loader_creator.create.assert_called_once_with(dataset=dataset, engine=None)


def test_dataset_data_source_has_data_loader_true() -> None:
    data_source = DatasetDataSource(
        datasets={},
        data_loader_creators={"train": AutoDataLoaderCreator(batch_size=4)},
    )
    assert data_source.has_data_loader("train")


def test_dataset_data_source_has_data_loader_false() -> None:
    data_source = DatasetDataSource(datasets={}, data_loader_creators={})
    assert not data_source.has_data_loader("missing")


def test_dataset_data_source_load_state_dict() -> None:
    data_source = DatasetDataSource(datasets={}, data_loader_creators={})
    data_source.load_state_dict({})


def test_dataset_data_source_state_dict() -> None:
    data_source = DatasetDataSource(datasets={}, data_loader_creators={})
    assert data_source.state_dict() == {}


def test_dataset_data_source_check_missing_dataset(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(level=logging.WARNING):
        DatasetDataSource(
            datasets={}, data_loader_creators={"train": Mock(spec=BaseDataLoaderCreator)}
        )
        assert len(caplog.messages) == 1


def test_dataset_data_source_check_missing_data_loader_creator(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(level=logging.WARNING):
        DatasetDataSource(datasets={"train": Mock(spec=Dataset)}, data_loader_creators={})
        assert len(caplog.messages) == 1
