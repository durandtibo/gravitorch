r"""This module defines the base class for the data loader collators."""

from __future__ import annotations

__all__ = ["BaseCollator", "setup_collator"]

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Generic, TypeVar

from objectory import AbstractFactory
from torch.utils.data.dataloader import default_collate

R = TypeVar("R")
T = TypeVar("T")


class BaseCollator(Generic[T, R], Callable[[list[T]], R], ABC, metaclass=AbstractFactory):
    r"""Defines the base class to create a batch of examples."""

    @abstractmethod
    def __call__(self, data: list[T]) -> R:
        r"""Creates a batch given a list of examples.

        Args:
        ----
            data (list): Specifies a list of examples.

        Returns:
        -------
             A batch of examples.
        """


def setup_collator(collator: Callable[[list[T]], R] | dict | None) -> Callable[[list[T]], R]:
    r"""Sets up a data loader collator.

    Args:
    ----
        collator (``Callable`` or dict or None): Specifies the
            data loader collator or its configuration. If ``None``,
            the default data loader collator is used.

    Returns:
    -------
        ``Callable``: The data loader collator.
    """
    if collator is None:
        collator = default_collate
    if isinstance(collator, dict):
        collator = BaseCollator.factory(**collator)
    return collator
