__all__ = ["BaseSysConfig"]

from abc import ABC

from objectory import AbstractFactory


class BaseSysConfig(ABC, metaclass=AbstractFactory):
    r"""Defines the base class to configure the system or some packages."""

    def configure(self) -> None:
        r"""Configure the system or a package."""

    def show(self) -> None:
        r"""Shows the information associated to the system or a package."""
