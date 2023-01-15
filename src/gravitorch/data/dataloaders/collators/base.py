r"""This module defines the base class for the data loader collators."""

__all__ = ["BaseCollator"]

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Generic, TypeVar

from objectory import AbstractFactory

R = TypeVar("R")
T = TypeVar("T")


class BaseCollator(Generic[T, R], Callable[[list[T]], R], ABC, metaclass=AbstractFactory):
    r"""Defines the base class to create a batch of examples."""

    @abstractmethod
    def __call__(self, data: list[T]) -> R:
        r"""Creates a batch given a list of examples.

        Args:
            data (list): Specifies a list of examples.

        Returns:
             A batch of examples.
        """
