r"""This module defines some path utility functions."""

__all__ = [
    "find_tar_files",
    "get_human_readable_file_size",
    "get_number_of_files",
    "get_original_cwd",
    "get_pythonpath",
    "sanitize_path",
    "working_directory",
]

import contextlib
import os
import tarfile
from pathlib import Path
from typing import Optional, Union
from urllib.parse import unquote, urlparse

import hydra
from hydra.core.hydra_config import HydraConfig

from gravitorch.utils.format import human_byte_size


def get_original_cwd() -> Path:
    r"""Gets the original working directory the experiment was launched from.

    The problem is that Hydra change the working directory when the
    application is launched.

    Returns:
        ``pathlib.Path``: If Hydra is initialized, it returns the
            original working directory otherwise it returns the
            current working directory.
    """
    if HydraConfig.initialized():
        return sanitize_path(hydra.utils.get_original_cwd())
    return Path.cwd()


def get_pythonpath() -> Path:
    r"""Gets the value of PYTHONPATH or the original working directory if this
    value is not defined.

    Returns:
        ``pathlib.Path``: The value of the PYTHONPATH or the original
            working directory if it is not defined.
    """
    return sanitize_path(os.environ.get("PYTHONPATH", get_original_cwd()))


@contextlib.contextmanager
def working_directory(path: Path):
    r"""A context manager which changes the working directory to the given path,
    and then changes it back to its previous value on exit.

    SOURCE: https://gist.github.com/nottrobin/3d675653244f8814838a

    Args:
        path (``pathlib.Path``): Specifies the path to the temporary
            working directory.

    Example usage:

    .. code-block:: python

        # Do something in original directory
        >>> with working_directory(Path('/my/new/path')):
        ...     # Do something in new directory
        # Back to old directory
    """
    path = sanitize_path(path)
    prev_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)


def get_number_of_files(path: str) -> int:
    r"""Gets the number of files in a folder and its sub-folders.

    Args:
        path (str): Specifies the path to the folder.

    Returns:
        int: The number of files.
    """
    return sum([len(files) for _, _, files in os.walk(path)])


def find_tar_files(path: Path, recursive: bool = True) -> tuple[Path, ...]:
    r"""Finds the path of all the tar files in a given path.

    This function does not check if a path is a symbolic link so be
    careful if you are using a path with symbolic links.

    Args:
        path (``pathlib.Path``): Specifies the path where to look for
            the tar files.
        recursive (bool, optional): Specifies if it should also check
            the sub-folders.

    Returns:
        tuple: The tuple of path of tar files.
    """
    path = sanitize_path(path)
    if path.is_dir():
        list_files = []
        for sub_path in path.iterdir():
            is_dir = sub_path.is_dir()
            if is_dir and recursive:
                list_files.extend(find_tar_files(sub_path))
            elif not is_dir and tarfile.is_tarfile(sub_path):
                list_files.append(sub_path)
        return tuple(list_files)
    if tarfile.is_tarfile(path):
        return (path,)
    return tuple()


def sanitize_path(path: Union[Path, str]) -> Path:
    r"""Sanitizes a given path.

    Args:
        path (``pathlib.Path`` or str): Specifies the path to
            sanitize.

    Returns:
        ``pathlib.Path``: The sanitized path.

    Example usage:

    .. code-block:: python

        # Let's assume the current path is /my/path
        >>> from pathlib import Path
        >>> from gravitorch.utils.path import sanitize_path
        >>> sanitize_path("something")
        PosixPath('/my/path/something')
        >>> sanitize_path("")
        PosixPath('/my/path')
        >>> sanitize_path(Path("something"))
        PosixPath('/my/path/something')
        >>> sanitize_path(Path("something/./../"))
        PosixPath('/my/path')
        # Support URI syntax
        >>> sanitize_path('file:///my/path/something/./../')
        PosixPath('/my/path')
    """
    if isinstance(path, str):
        # Use urlparse to parse file URI: https://stackoverflow.com/a/15048213
        path = Path(unquote(urlparse(path).path))
    return path.expanduser().resolve()


def get_human_readable_file_size(path: Union[Path, str], unit: Optional[str] = None) -> str:
    r"""Gets a human-readable representation of a file size.

    Args:
        path (``pathlib.Path`` or str): Specifies the file.
        unit (str, optional): Specifies the unit. If ``None``, the
            best unit is found automatically. The supported units
            are: ``'B'``, ``'KB'``, ``'MB'``, ``'GB'``, ``'TB'``.
            Default: ``None``

    Returns:
        str: The file size in a human-readable format.

    Example usage:

    .. code-block:: python

        >>> from gravitorch.utils.path import get_human_readable_file_size
        >>> get_human_readable_file_size('/my/path/data.txt')
        2.00 KB
    """
    return human_byte_size(size=sanitize_path(path).stat().st_size, unit=unit)
