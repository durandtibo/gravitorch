__all__ = ["PickleSaverIterDataPipe", "PyTorchSaverIterDataPipe"]

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Union

from torch.utils.data import IterDataPipe

from gravitorch.utils.format import str_indent
from gravitorch.utils.io import save_pickle, save_pytorch
from gravitorch.utils.path import sanitize_path

logger = logging.getLogger(__name__)


class PickleSaverIterDataPipe(IterDataPipe[Path]):
    r"""Implements a DataPipe to save each value from the source DataPipe in a
    pickle file.

    This DataPipe returns the path to each pickle file. If the source
    DataPipe has ``M`` values, ``M`` pickle files are created.

    Args:
        datapipe: Specifies the source DataPipe.
        root_path (``pathlib.Path`` or str): Specifies the directory
            where to save the pickle files.
        pattern (str, optional): Specifies the filename pattern of
            the file. The pattern should have ``'{index}'`` or similar
            syntax to indicate the index of the file.
            Default: ``'data_{index:04d}.pkl'``
    """

    def __init__(
        self,
        datapipe: IterDataPipe,
        root_path: Union[Path, str],
        pattern: str = "data_{index:04d}.pkl",
    ):
        self._datapipe = datapipe
        self._root_path = sanitize_path(root_path)
        if not self._root_path.is_dir():
            raise ValueError(f"root_path has to be a directory (received: {self._root_path})")
        self._pattern = str(pattern)
        if "index" not in self._pattern:
            raise ValueError(f"pattern does not have 'index' (received: {self._pattern})")

    def __iter__(self) -> Iterator[Path]:
        for i, data in enumerate(self._datapipe):
            path = self._root_path.joinpath(self._pattern.format(index=i))
            save_pickle(data, path)
            yield path

    def __len__(self) -> int:
        return len(self._datapipe)

    def __str__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  root_path={self._root_path},\n"
            f"  pattern={self._pattern},\n"
            f"  datapipe={str_indent(self._datapipe)},\n)"
        )


class PyTorchSaverIterDataPipe(IterDataPipe[Path]):
    r"""Implements a DataPipe to save each value from the source DataPipe in a
    PyTorch file.

    This DataPipe returns the path to each PyTorch file. If the source
    DataPipe has ``M`` values, ``M`` PyTorch files are created.

    Args:
        datapipe: Specifies the source DataPipe.
        root_path (``pathlib.Path`` or str): Specifies the directory
            where to save the PyTorch files.
        pattern (str, optional): Specifies the filename pattern of the
            file. The pattern should have ``'{index}'`` or similar
            syntax to indicate the index of the file.
            Default: ``'data_{index:04d}.pt'``
    """

    def __init__(
        self,
        datapipe: IterDataPipe,
        root_path: Union[Path, str],
        pattern: str = "data_{index:04d}.pt",
    ):
        self._datapipe = datapipe
        self._root_path = sanitize_path(root_path)
        if not self._root_path.is_dir():
            raise ValueError(f"root_path has to be a directory (received: {self._root_path})")
        self._pattern = str(pattern)
        if "index" not in self._pattern:
            raise ValueError(f"pattern does not have 'index' (received: {self._pattern})")

    def __iter__(self) -> Iterator[Path]:
        for i, data in enumerate(self._datapipe):
            path = self._root_path.joinpath(self._pattern.format(index=i))
            save_pytorch(data, path)
            yield path

    def __len__(self) -> int:
        return len(self._datapipe)

    def __str__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  root_path={self._root_path},\n"
            f"  pattern={self._pattern},\n"
            f"  datapipe={str_indent(self._datapipe)},\n)"
        )
