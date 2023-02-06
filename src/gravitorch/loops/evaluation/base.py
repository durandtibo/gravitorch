r"""This module defines the base class to implement evaluation loops."""
__all__ = ["BaseEvaluationLoop"]

import logging
from abc import ABC, abstractmethod
from typing import Any

from objectory import AbstractFactory

from gravitorch.engines.base import BaseEngine

logger = logging.getLogger(__name__)


class BaseEvaluationLoop(ABC, metaclass=AbstractFactory):
    r"""Defines the evaluation loop base class.

    To implement your own evaluation loop, you will need to define the
    following methods:

        - ``eval``
        - ``load_state_dict``
        - ``state_dict``
    """

    @abstractmethod
    def eval(self, engine: BaseEngine) -> None:
        r"""Evaluates the model on the evaluation dataset.

        Args:
            engine (``BaseEngine``): Specifies the engine.
        """

    @abstractmethod
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        r"""Sets up the evaluation loop from a dictionary containing the state
        values.

        Args:
            state_dict (dict): Specifies a dictionary
                containing state keys with values.
        """

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        r"""Returns a dictionary containing state values.

        Returns:
            dict: The state values in a dict.
        """
