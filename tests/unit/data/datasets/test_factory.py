from coola import objects_are_equal
from objectory import OBJECT_TARGET

from gravitorch.data.datasets import (
    DummyMultiClassDataset,
    ExampleDataset,
    create_datasets,
    setup_dataset,
)

#####################################
#     Tests for create_datasets     #
#####################################


def test_create_datasets_empty() -> None:
    assert create_datasets({}) == {}


def test_create_datasets_one() -> None:
    assert objects_are_equal(
        create_datasets(
            {
                "train": {
                    "_target_": "gravitorch.data.datasets.ExampleDataset",
                    "examples": (1, 2, 3),
                }
            }
        ),
        {"train": ExampleDataset((1, 2, 3))},
    )


###################################
#     Tests for setup_dataset     #
###################################


def test_setup_dataset_object() -> None:
    dataset = DummyMultiClassDataset(num_examples=10, num_classes=5)
    assert setup_dataset(dataset) is dataset


def test_setup_dataset_dict() -> None:
    assert isinstance(
        setup_dataset(
            {
                OBJECT_TARGET: "gravitorch.data.datasets.DummyMultiClassDataset",
                "num_examples": 10,
                "num_classes": 5,
            }
        ),
        DummyMultiClassDataset,
    )


def test_setup_dataset_none() -> None:
    assert setup_dataset(None) is None
