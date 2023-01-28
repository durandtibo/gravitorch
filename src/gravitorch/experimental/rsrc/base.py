__all__ = ["BaseResourceManager"]

from abc import abstractmethod
from contextlib import AbstractContextManager

from objectory import AbstractFactory


class BaseResourceManager(AbstractContextManager, metaclass=AbstractFactory):
    r"""Defines the base class to manage a resource."""

    @abstractmethod
    def configure(self) -> None:
        r"""Configure the resource."""

    @abstractmethod
    def show(self) -> None:
        r"""Shows the information associated to resource."""
