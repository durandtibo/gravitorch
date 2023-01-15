__all__ = ["PyTorchArtifact"]

from pathlib import Path
from typing import Any

from gravitorch.utils.artifacts.base import BaseArtifact
from gravitorch.utils.io import save_pytorch


class PyTorchArtifact(BaseArtifact):
    r"""Implements a PyTorch artifact.

    Args:
        tag (str): Specifies the artifact tag. The tag is used to
            define the PyTorch filename.
        data: Specifies the data to save in the PyTorch artifact. The
            data should be PyTorch compatible.
    """

    def __init__(self, tag: str, data: Any):
        self._tag = str(tag)
        self._data = data

    def __str__(self) -> str:
        return f"{self.__class__.__qualname__}(tag={self._tag})"

    def create(self, path: Path) -> None:
        save_pytorch(to_save=self._data, path=path.joinpath(f"{self._tag}.pt"))
