r"""This package contains the implementation of some datasets."""

__all__ = [
    "DemoMultiClassClsDataset",
    "FileToInMemoryDataset",
    "ImageFolderDataset",
    "InMemoryDataset",
    "MNISTDataset",
    "log_box_dataset_class",
    "setup_dataset",
]

from gravitorch.data.datasets.demo_map_style import DemoMultiClassClsDataset
from gravitorch.data.datasets.factory import setup_dataset
from gravitorch.data.datasets.image_folder import ImageFolderDataset
from gravitorch.data.datasets.in_memory import FileToInMemoryDataset, InMemoryDataset
from gravitorch.data.datasets.mnist import MNISTDataset
from gravitorch.data.datasets.utils import log_box_dataset_class
