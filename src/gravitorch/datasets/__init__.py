r"""This package contains the implementation of some datasets."""

from __future__ import annotations

__all__ = [
    "DummyMultiClassDataset",
    "ExampleDataset",
    "create_datasets",
    "is_dataset_config",
    "log_box_dataset_class",
    "setup_dataset",
]

from gravitorch.datasets.dummy import DummyMultiClassDataset
from gravitorch.datasets.example import ExampleDataset
from gravitorch.datasets.factory import (
    create_datasets,
    is_dataset_config,
    setup_dataset,
)
from gravitorch.datasets.utils import log_box_dataset_class
