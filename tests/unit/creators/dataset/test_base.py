from __future__ import annotations

from objectory import OBJECT_TARGET

from gravitorch.creators.dataset import DatasetCreator, setup_dataset_creator
from gravitorch.datasets import ExampleDataset

###########################################
#     Tests for setup_dataset_creator     #
###########################################


def test_setup_dataset_creator_object() -> None:
    creator = DatasetCreator(ExampleDataset((1, 2, 3, 4, 5)))
    assert setup_dataset_creator(creator) is creator


def test_setup_dataset_creator_dict() -> None:
    assert isinstance(
        setup_dataset_creator(
            {
                OBJECT_TARGET: "gravitorch.creators.dataset.DatasetCreator",
                "dataset": {
                    OBJECT_TARGET: "gravitorch.datasets.ExampleDataset",
                    "examples": (1, 2, 3, 4, 5),
                },
            },
        ),
        DatasetCreator,
    )
