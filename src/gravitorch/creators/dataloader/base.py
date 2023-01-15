r"""This module defines the base class for the data loader creators."""

from __future__ import annotations

__all__ = ["BaseDataLoaderCreator"]

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING, Generic, TypeVar

from objectory import AbstractFactory
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from gravitorch.engines import BaseEngine

T = TypeVar("T")


class BaseDataLoaderCreator(Generic[T], ABC, metaclass=AbstractFactory):
    r"""Defines the base class to create data loader.

    This class and its child classes are designed to be used in a data
    source.
    """

    @abstractmethod
    def create(self, dataset: Dataset, engine: BaseEngine | None = None) -> Iterable[T]:
        r"""Creates a data loader given a dataset and an engine.

        The engine can be used to get the epoch, or other information
        about the training/evaluation.

        Args:
            dataset (``torch.utils.data.Dataset``): Specifies the
                dataset.
            engine (``gravitorch.engines.BaseEngine`` or ``None``,
                optional): Specifies an engine. Default: ``None``

        Returns:
            ``Iterable``: The instantiated data loader.
        """
