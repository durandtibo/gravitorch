__all__ = ["BaseRunner"]

from abc import ABC, abstractmethod
from typing import Any

from objectory import AbstractFactory


class BaseRunner(ABC, metaclass=AbstractFactory):
    r"""Defines the base class of the runners."""

    @abstractmethod
    def run(self) -> Any:
        r"""Executes the logic of the runner.

        Returns:
            Any artifact of the runner
        """
