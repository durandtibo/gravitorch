from __future__ import annotations

__all__ = ["BaseRunner"]

from abc import ABC, abstractmethod
from typing import Any

from objectory import AbstractFactory


class BaseRunner(ABC, metaclass=AbstractFactory):
    r"""Defines the base class of the runners.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.runners import TrainingRunner
        >>> from gravitorch.testing import create_dummy_engine
        >>> engine = create_dummy_engine()
        >>> runner = TrainingRunner(engine)
        >>> runner.run()
    """

    @abstractmethod
    def run(self) -> Any:
        r"""Executes the logic of the runner.

        Returns
        -------
            Any artifact of the runner

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.runners import TrainingRunner
            >>> from gravitorch.testing import create_dummy_engine
            >>> runner = TrainingRunner(create_dummy_engine())
            >>> engine = runner.run()
        """
