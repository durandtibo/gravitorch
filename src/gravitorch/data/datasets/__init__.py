r"""This package contains the implementation of some datasets."""

from __future__ import annotations

__all__ = [
    "DummyMultiClassDataset",
    "FileToInMemoryDataset",
    "ImageFolderDataset",
    "InMemoryDataset",
    "MNIST",
    "log_box_dataset_class",
    "setup_dataset",
]

from gravitorch.data.datasets.dummy import DummyMultiClassDataset
from gravitorch.data.datasets.factory import setup_dataset
from gravitorch.data.datasets.image_folder import ImageFolderDataset
from gravitorch.data.datasets.in_memory import FileToInMemoryDataset, InMemoryDataset
from gravitorch.data.datasets.mnist import MNIST
from gravitorch.data.datasets.utils import log_box_dataset_class
