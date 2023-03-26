__all__ = ["BaseBatchLoader"]

from collections.abc import Iterable
from contextlib import AbstractContextManager
from typing import TypeVar

T = TypeVar("T")


class BaseBatchLoader(Iterable[T], AbstractContextManager):
    r"""Defines the base class to implement batch loader."""
