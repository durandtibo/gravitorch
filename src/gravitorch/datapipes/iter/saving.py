from __future__ import annotations

__all__ = ["PickleSaverIterDataPipe", "PyTorchSaverIterDataPipe"]

import logging
from collections.abc import Iterator
from pathlib import Path

from coola.utils import str_indent, str_mapping
from torch.utils.data import IterDataPipe

from gravitorch.utils.io import save_pickle, save_pytorch
from gravitorch.utils.path import sanitize_path

logger = logging.getLogger(__name__)


class PickleSaverIterDataPipe(IterDataPipe[Path]):
    r"""Implements a DataPipe to save each value from the source DataPipe
    in a pickle file.

    This DataPipe returns the path to each pickle file. If the source
    DataPipe has ``M`` values, ``M`` pickle files are created.

    Args:
    ----
        datapipe: Specifies the source DataPipe.
        root_path (``pathlib.Path`` or str): Specifies the directory
            where to save the pickle files.
        pattern (str, optional): Specifies the filename pattern of
            the file. The pattern should have ``'{index}'`` or similar
            syntax to indicate the index of the file.
            Default: ``'data_{index:04d}.pkl'``

    Example usage:

    .. code-block:: pycon

        >>> from torch.utils.data.datapipes.iter import IterableWrapper
        >>> from gravitorch.datapipes.iter import PickleSaver
        >>> dp = PickleSaver(IterableWrapper([1, 2, 3, 4, 5]), root_path="tmp/")
        >>> dp
        PickleSaverIterDataPipe(
          (root_path): .../tmp
          (pattern): data_{index:04d}.pkl
          (datapipe): IterableWrapperIterDataPipe
        )
    """

    def __init__(
        self,
        datapipe: IterDataPipe,
        root_path: Path | str,
        pattern: str = "data_{index:04d}.pkl",
    ) -> None:
        self._datapipe = datapipe
        self._root_path = sanitize_path(root_path)
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

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        args = str_indent(
            str_mapping(
                {
                    "root_path": self._root_path,
                    "pattern": self._pattern,
                    "datapipe": self._datapipe,
                }
            )
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"


class PyTorchSaverIterDataPipe(IterDataPipe[Path]):
    r"""Implements a DataPipe to save each value from the source DataPipe
    in a PyTorch file.

    This DataPipe returns the path to each PyTorch file. If the source
    DataPipe has ``M`` values, ``M`` PyTorch files are created.

    Args:
    ----
        datapipe: Specifies the source DataPipe.
        root_path (``pathlib.Path`` or str): Specifies the directory
            where to save the PyTorch files.
        pattern (str, optional): Specifies the filename pattern of the
            file. The pattern should have ``'{index}'`` or similar
            syntax to indicate the index of the file.
            Default: ``'data_{index:04d}.pt'``

    Example usage:

    .. code-block:: pycon

        >>> from torch.utils.data.datapipes.iter import IterableWrapper
        >>> from gravitorch.datapipes.iter import PyTorchSaver
        >>> dp = PyTorchSaver(IterableWrapper([1, 2, 3, 4, 5]), root_path="tmp/")
        >>> dp
        PyTorchSaverIterDataPipe(
          (root_path): .../tmp
          (pattern): data_{index:04d}.pt
          (datapipe): IterableWrapperIterDataPipe
        )
    """

    def __init__(
        self,
        datapipe: IterDataPipe,
        root_path: Path | str,
        pattern: str = "data_{index:04d}.pt",
    ) -> None:
        self._datapipe = datapipe
        self._root_path = sanitize_path(root_path)
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

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        args = str_indent(
            str_mapping(
                {
                    "root_path": self._root_path,
                    "pattern": self._pattern,
                    "datapipe": self._datapipe,
                }
            )
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"
