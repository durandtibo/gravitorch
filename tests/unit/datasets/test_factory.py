from coola import objects_are_equal
from objectory import OBJECT_TARGET

from gravitorch.datasets import (
    DummyMultiClassDataset,
    ExampleDataset,
    create_datasets,
    is_dataset_config,
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
                    OBJECT_TARGET: "gravitorch.datasets.ExampleDataset",
                    "examples": (1, 2, 3),
                }
            }
        ),
        {"train": ExampleDataset((1, 2, 3))},
    )


def test_create_datasets_two() -> None:
    assert objects_are_equal(
        create_datasets(
            {
                "train": {
                    OBJECT_TARGET: "gravitorch.datasets.ExampleDataset",
                    "examples": (1, 2, 3),
                },
                "val": ExampleDataset((4, 5)),
            }
        ),
        {"train": ExampleDataset((1, 2, 3)), "val": ExampleDataset((4, 5))},
    )


def test_create_datasets_three() -> None:
    assert objects_are_equal(
        create_datasets(
            {
                "train": {
                    OBJECT_TARGET: "gravitorch.datasets.ExampleDataset",
                    "examples": (1, 2, 3),
                },
                "val": ExampleDataset((4, 5)),
                "test": {
                    OBJECT_TARGET: "gravitorch.datasets.ExampleDataset",
                    "examples": ("A", "B"),
                },
            }
        ),
        {
            "train": ExampleDataset((1, 2, 3)),
            "val": ExampleDataset((4, 5)),
            "test": ExampleDataset(("A", "B")),
        },
    )


#######################################
#     Tests for is_dataset_config     #
#######################################


def test_is_dataset_config_true() -> None:
    assert is_dataset_config(
        {OBJECT_TARGET: "gravitorch.datasets.ExampleDataset", "examples": ("A", "B")}
    )


def test_is_dataset_config_false() -> None:
    assert not is_dataset_config({OBJECT_TARGET: "torch.nn.Identity"})


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
                OBJECT_TARGET: "gravitorch.datasets.DummyMultiClassDataset",
                "num_examples": 10,
                "num_classes": 5,
            }
        ),
        DummyMultiClassDataset,
    )


def test_setup_dataset_none() -> None:
    assert setup_dataset(None) is None
