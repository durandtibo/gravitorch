from __future__ import annotations

__all__ = ["DirFilterIterDataPipe", "FileFilterIterDataPipe", "PathListerIterDataPipe"]

from collections.abc import Iterator
from pathlib import Path

from coola.utils import str_indent, str_mapping
from torch.utils.data import IterDataPipe


class DirFilterIterDataPipe(IterDataPipe[Path]):
    r"""Implements an ``IterDataPipe`` to keep only the directory.

    Args:
    ----
        datapipe (``IterDataPipe``): Specifies the source
            ``IterDataPipe``.

    Example usage:

    .. code-block:: pycon

        >>> from pathlib import Path
        >>> from torch.utils.data.datapipes.iter import IterableWrapper
        >>> from gravitorch.datapipes.iter import DirFilter
        >>> dp = DirFilter(IterableWrapper([Path("tmp/A"), Path("tmp/B"), Path("tmp/C")]))
        >>> dp
        DirFilterIterDataPipe(
          (datapipe): IterableWrapperIterDataPipe
        )
    """

    def __init__(self, datapipe: IterDataPipe[Path]) -> None:
        self._datapipe = datapipe

    def __iter__(self) -> Iterator[Path]:
        for path in self._datapipe:
            if path.is_dir():
                yield path

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        args = str_indent(str_mapping({"datapipe": self._datapipe}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"


class FileFilterIterDataPipe(IterDataPipe[Path]):
    r"""Implements an ``IterDataPipe`` to keep only the files.

    Args:
    ----
        datapipe (``IterDataPipe``): Specifies the source
            ``IterDataPipe``.

    Example usage:

    .. code-block:: pycon

        >>> from pathlib import Path
        >>> from torch.utils.data.datapipes.iter import IterableWrapper
        >>> from gravitorch.datapipes.iter import FileFilter
        >>> dp = FileFilter(IterableWrapper([Path("tmp/A"), Path("tmp/B"), Path("tmp/C")]))
        >>> dp
        FileFilterIterDataPipe(
          (datapipe): IterableWrapperIterDataPipe
        )
    """

    def __init__(self, datapipe: IterDataPipe[Path]) -> None:
        self._datapipe = datapipe

    def __iter__(self) -> Iterator[Path]:
        for path in self._datapipe:
            if path.is_file():
                yield path

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        args = str_indent(str_mapping({"datapipe": self._datapipe}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"


class PathListerIterDataPipe(IterDataPipe[Path]):
    r"""Implements an ``IterDataPipe`` to list the paths.

    Args:
    ----
        datapipe (``IterDataPipe``): Specifies the source
            ``IterDataPipe`` with the root paths.
        pattern (str, optional): Specifies a glob pattern, to return
            only the matching paths. Default: ``'*'``
        deterministic (bool, optional): If ``True``, the paths are
            returned in a deterministic order. Default: ``True``

    Example usage:

    .. code-block:: pycon

        >>> from pathlib import Path
        >>> from torch.utils.data.datapipes.iter import IterableWrapper
        >>> from gravitorch.datapipes.iter import PathLister
        >>> dp = PathLister(IterableWrapper([Path("tmp/A"), Path("tmp/B"), Path("tmp/C")]))
        >>> dp
        PathListerIterDataPipe(
          (pattern): *
          (deterministic): True
          (datapipe): IterableWrapperIterDataPipe
        )
    """

    def __init__(
        self,
        datapipe: IterDataPipe[Path],
        pattern: str = "*",
        deterministic: bool = True,
    ) -> None:
        self._datapipe = datapipe
        self._pattern = pattern
        self._deterministic = bool(deterministic)

    def __iter__(self) -> Iterator[Path]:
        for path in self._datapipe:
            paths = path.glob(self._pattern)
            if self._deterministic:
                paths = sorted(paths)
            yield from paths

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        args = str_indent(
            str_mapping(
                {
                    "pattern": self._pattern,
                    "deterministic": self._deterministic,
                    "datapipe": self._datapipe,
                }
            )
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"
