__all__ = ["DirFilterIterDataPipe", "FileFilterIterDataPipe", "PathListerIterDataPipe"]

from collections.abc import Iterator
from pathlib import Path

from torch.utils.data import IterDataPipe

from gravitorch.utils.format import str_indent


class DirFilterIterDataPipe(IterDataPipe[Path]):
    r"""Implements an ``IterDataPipe`` to keep only the directory.

    Args:
        datapipe (``IterDataPipe``): Specifies the source
            ``IterDataPipe``.
    """

    def __init__(self, datapipe: IterDataPipe[Path]):
        self._datapipe = datapipe

    def __iter__(self) -> Iterator[Path]:
        for path in self._datapipe:
            if path.is_dir():
                yield path

    def __str__(self) -> str:
        return f"{self.__class__.__qualname__}(\n" f"  datapipe={str_indent(self._datapipe)},\n)"


class FileFilterIterDataPipe(IterDataPipe[Path]):
    r"""Implements an ``IterDataPipe`` to keep only the files.

    Args:
        datapipe (``IterDataPipe``): Specifies the source
            ``IterDataPipe``.
    """

    def __init__(self, datapipe: IterDataPipe[Path]):
        self._datapipe = datapipe

    def __iter__(self) -> Iterator[Path]:
        for path in self._datapipe:
            if path.is_file():
                yield path

    def __str__(self) -> str:
        return f"{self.__class__.__qualname__}(\n" f"  datapipe={str_indent(self._datapipe)},\n)"


class PathListerIterDataPipe(IterDataPipe[Path]):
    r"""Implements an ``IterDataPipe`` to list the paths.

    Args:
        datapipe (``IterDataPipe``): Specifies the source
            ``IterDataPipe`` with the root paths.
        pattern (str, optional): Specifies a glob pattern, to return
            only the matching paths. Default: ``'*'``
        deterministic (bool, optional): If ``True``, the paths are
            returned in a deterministic order. Default: ``True``
    """

    def __init__(
        self,
        datapipe: IterDataPipe[Path],
        pattern: str = "*",
        deterministic: bool = True,
    ):
        self._datapipe = datapipe
        self._pattern = pattern
        self._deterministic = bool(deterministic)

    def __iter__(self) -> Iterator[Path]:
        for path in self._datapipe:
            paths = path.glob(self._pattern)
            if self._deterministic:
                paths = sorted(paths)
            yield from paths

    def __str__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  pattern={self._pattern},\n"
            f"  deterministic={self._deterministic},\n"
            f"  datapipe={str_indent(self._datapipe)},\n)"
        )
