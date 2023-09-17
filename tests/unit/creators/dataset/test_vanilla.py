from __future__ import annotations

from unittest.mock import Mock

from objectory import OBJECT_TARGET
from torch.utils.data import Dataset

from gravitorch.creators.dataset import DatasetCreator
from gravitorch.data.datasets import DummyMultiClassDataset

####################################
#     Tests for DatasetCreator     #
####################################


def test_dataset_creator_str() -> None:
    assert str(DatasetCreator(Mock(spec=Dataset))).startswith("DatasetCreator(")


def test_dataset_creator_create_dataset_object() -> None:
    dataset = Mock(spec=Dataset)
    assert DatasetCreator(dataset).create() is dataset


def test_dataset_creator_create_dataset_config() -> None:
    assert isinstance(
        DatasetCreator(
            {
                OBJECT_TARGET: "gravitorch.data.datasets.DummyMultiClassDataset",
                "num_examples": 10,
                "num_classes": 2,
                "feature_size": 4,
            }
        ).create(),
        DummyMultiClassDataset,
    )


def test_dataset_creator_create_dataset_cache_true() -> None:
    creator = DatasetCreator(
        {
            OBJECT_TARGET: "gravitorch.data.datasets.DummyMultiClassDataset",
            "num_examples": 10,
            "num_classes": 2,
            "feature_size": 4,
        },
        cache=True,
    )
    dataset1 = creator.create()
    dataset2 = creator.create()
    assert isinstance(dataset1, DummyMultiClassDataset)
    assert isinstance(dataset2, DummyMultiClassDataset)
    assert dataset1 is dataset2


def test_dataset_creator_create_dataset_cache_false() -> None:
    creator = DatasetCreator(
        {
            OBJECT_TARGET: "gravitorch.data.datasets.DummyMultiClassDataset",
            "num_examples": 10,
            "num_classes": 2,
            "feature_size": 4,
        },
    )
    dataset1 = creator.create()
    dataset2 = creator.create()
    assert isinstance(dataset1, DummyMultiClassDataset)
    assert isinstance(dataset2, DummyMultiClassDataset)
    assert dataset1 is not dataset2
