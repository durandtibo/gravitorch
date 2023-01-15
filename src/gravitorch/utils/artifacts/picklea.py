__all__ = ["PickleArtifact"]

from pathlib import Path
from pickle import HIGHEST_PROTOCOL
from typing import Any

from gravitorch.utils.artifacts.base import BaseArtifact
from gravitorch.utils.io import save_pickle


class PickleArtifact(BaseArtifact):
    r"""Implements a Pickle artifact.

    Args:
        tag (str): Specifies the artifact tag. The tag is used to
            define the Pickle filename.
        data: Specifies the data to save in the Pickle artifact. The
            data should be Pickle compatible.
        protocol (int, optional): Specifies the pickle protocol.
            Default: highest protocol available.
    """

    def __init__(self, tag: str, data: Any, protocol: int = HIGHEST_PROTOCOL):
        self._tag = str(tag)
        self._data = data
        self._protocol = protocol

    def __str__(self) -> str:
        return f"{self.__class__.__qualname__}(tag={self._tag}, protocol={self._protocol})"

    def create(self, path: Path) -> None:
        save_pickle(
            to_save=self._data, path=path.joinpath(f"{self._tag}.pkl"), protocol=self._protocol
        )
