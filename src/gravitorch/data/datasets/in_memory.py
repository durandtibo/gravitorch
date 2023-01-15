r"""This module implements datasets that stores all the examples in memory."""

__all__ = ["InMemoryDataset", "FileToInMemoryDataset"]

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TypeVar, Union

import torch
from torch.utils.data import Dataset

from gravitorch.data.datasets.utils import log_box_dataset_class
from gravitorch.utils.io import load_json, load_pickle
from gravitorch.utils.path import sanitize_path

logger = logging.getLogger(__name__)

T = TypeVar("T")


class InMemoryDataset(Dataset[T]):
    r"""Implements a dataset that stores all the examples in-memory.

    You can use this dataset only if all the examples can fit
    in-memory.

    Args:
        examples: Specifies the examples of the dataset.
    """

    def __init__(self, examples: Sequence[T]):
        log_box_dataset_class(self)
        self._examples = tuple(examples)

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, item: int) -> T:
        return self._examples[item]

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(num_examples={len(self):,})"


class FileToInMemoryDataset(Dataset[T]):
    r"""Implements a dataset that loads examples from a file and store them in
    memory.

    The data in the files should already be preprocessed and organized
    by examples. The file should store a tuple (or list) of examples.
    You can use this dataset only if all the examples can fit
    in-memory.

    This dataset supports the following file formats:

        - PyTorch (file created by ``torch.save``): the extension of
            the file has to be ``.pt``
        - pickle: the extension of the file has to be ``.pkl``
        - json: the extension of the file has to be ``.json``

    The extension of the file is used to find the loader to use.

    Args:
        path (``pathlib.Path`` or str): Specifies the path to the file
            to load.
    """

    def __init__(self, path: Union[Path, str]):
        log_box_dataset_class(self)
        self._path = sanitize_path(path)
        logger.info(f"Loading data from: {self._path}")
        self._examples = _load_examples(self._path)

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, item: int) -> T:
        return self._examples[item]

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(num_examples={len(self):,}, path={self._path})"


def _load_examples(path: Path) -> tuple[T, ...]:
    r"""Loads the examples from a file.

    This function supports the following file formats:

        - PyTorch (file created by ``torch.save``): the extension of
            the file has to be ``.pt``
        - pickle: the extension of the file has to be ``.pkl``
        - json: the extension of the file has to be ``.json``

    Args:
        path (``pathlib.Path`` or str): Specifies the path to the file
            to load.
    """
    if path.suffix == ".pt":
        return tuple(torch.load(path))
    if path.suffix == ".pkl":
        return tuple(load_pickle(path))
    if path.suffix == ".json":
        return tuple(load_json(path))
    raise ValueError(
        f"Incorrect file extension {path.suffix}. The supported file extensions "
        "are '.pt', '.pkl' and '.json'"
    )
