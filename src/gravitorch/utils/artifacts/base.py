__all__ = ["BaseArtifact"]

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, TypeVar

T = TypeVar("T")


class BaseArtifact(Generic[T], ABC):
    r"""Defines the base class to implement an artifact."""

    @abstractmethod
    def create(self, path: Path) -> None:
        r"""Creates the artifact.

        Args:
            path (``pathlib.Path``): Specifies the path where to write
                the artifact.
        """
