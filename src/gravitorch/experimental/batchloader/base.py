__all__ = ["BaseBatchLoader"]

from collections.abc import Iterable
from contextlib import AbstractContextManager
from typing import TypeVar

from objectory import AbstractFactory

T = TypeVar("T")


class BaseBatchLoader(Iterable[T], AbstractContextManager, metaclass=AbstractFactory):
    r"""Defines the base class to implement batch loader."""
