from objectory import OBJECT_TARGET

from gravitorch.data.datasets import DummyMultiClassDataset, setup_dataset

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
