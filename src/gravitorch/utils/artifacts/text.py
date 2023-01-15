__all__ = ["TextArtifact"]

from pathlib import Path

from gravitorch.utils.artifacts.base import BaseArtifact
from gravitorch.utils.io import save_text


class TextArtifact(BaseArtifact):
    r"""Implements a text artifact.

    Args:
        tag (str): Specifies the artifact tag. The tag is used to
            define the text filename.
        data (str): Specifies the data to save in the text artifact.
    """

    def __init__(self, tag: str, data: str):
        self._tag = str(tag)
        self._data = str(data)

    def __str__(self) -> str:
        return f"{self.__class__.__qualname__}(tag={self._tag})"

    def create(self, path: Path) -> None:
        save_text(to_save=self._data, path=path.joinpath(f"{self._tag}.txt"))
