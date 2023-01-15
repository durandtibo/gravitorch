from objectory import OBJECT_TARGET

from gravitorch.data.datasets import DemoMultiClassClsDataset, setup_dataset

###################################
#     Tests for setup_dataset     #
###################################


def test_setup_dataset_object():
    dataset = DemoMultiClassClsDataset(num_examples=10, num_classes=5)
    assert setup_dataset(dataset) is dataset


def test_setup_dataset_dict():
    assert isinstance(
        setup_dataset(
            {
                OBJECT_TARGET: "gravitorch.data.datasets.DemoMultiClassClsDataset",
                "num_examples": 10,
                "num_classes": 5,
            }
        ),
        DemoMultiClassClsDataset,
    )


def test_setup_dataset_none():
    assert setup_dataset(None) is None
