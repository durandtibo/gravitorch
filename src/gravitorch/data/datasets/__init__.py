r"""This package contains the implementation of some datasets."""

from __future__ import annotations

__all__ = [
    "DummyMultiClassDataset",
    "ImageFolderDataset",
    "ExampleDataset",
    "create_datasets",
    "is_dataset_config",
    "log_box_dataset_class",
    "setup_dataset",
]

from gravitorch.data.datasets.dummy import DummyMultiClassDataset
from gravitorch.data.datasets.example import ExampleDataset
from gravitorch.data.datasets.factory import (
    create_datasets,
    is_dataset_config,
    setup_dataset,
)
from gravitorch.data.datasets.image_folder import ImageFolderDataset
from gravitorch.data.datasets.utils import log_box_dataset_class
