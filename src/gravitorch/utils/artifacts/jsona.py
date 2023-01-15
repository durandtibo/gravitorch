__all__ = ["JSONArtifact"]

from pathlib import Path
from typing import Any

from gravitorch.utils.artifacts.base import BaseArtifact
from gravitorch.utils.io import save_json


class JSONArtifact(BaseArtifact):
    r"""Implements a JSON artifact.

    Args:
        tag (str): Specifies the artifact tag. The tag is used to
            define the JSON filename.
        data: Specifies the data to save in the JSON artifact. The data
            should be JSON compatible.
    """

    def __init__(self, tag: str, data: Any):
        self._tag = str(tag)
        self._data = data

    def __str__(self) -> str:
        return f"{self.__class__.__qualname__}(tag={self._tag})"

    def create(self, path: Path) -> None:
        save_json(to_save=self._data, path=path.joinpath(f"{self._tag}.json"))
