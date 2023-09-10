r"""This module defines the base class to implement evaluation loops."""

from __future__ import annotations

__all__ = ["BaseEvaluationLoop"]

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from objectory import AbstractFactory

if TYPE_CHECKING:
    from gravitorch.engines import BaseEngine

logger = logging.getLogger(__name__)


class BaseEvaluationLoop(ABC, metaclass=AbstractFactory):
    r"""Defines the evaluation loop base class.

    To implement your own evaluation loop, you will need to define the
    following methods:

        - ``eval``
        - ``load_state_dict``
        - ``state_dict``

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.loops.evaluation import EvaluationLoop
        >>> from gravitorch.testing import create_dummy_engine
        >>> engine = create_dummy_engine()
        >>> loop = EvaluationLoop()
        >>> loop
        EvaluationLoop(
          (batch_device_placement): AutoDevicePlacement(device=cpu)
          (grad_enabled): False
          (tag): eval
          (condition): EveryEpochEvalCondition(every=1)
          (observer): NoOpLoopObserver()
          (profiler): NoOpProfiler()
        )
        >>> loop.eval(engine)
    """

    @abstractmethod
    def eval(self, engine: BaseEngine) -> None:
        r"""Evaluates the model on the evaluation dataset.

        Args:
        ----
            engine (``BaseEngine``): Specifies the engine.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.loops.evaluation import EvaluationLoop
            >>> from gravitorch.testing import create_dummy_engine
            >>> engine = create_dummy_engine()
            >>> loop = EvaluationLoop()
            >>> loop.eval(engine)
        """

    @abstractmethod
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        r"""Sets up the evaluation loop from a dictionary containing the
        state values.

        Args:
        ----
            state_dict (dict): Specifies a dictionary
                containing state keys with values.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.loops.evaluation import EvaluationLoop
            >>> loop = EvaluationLoop()
            >>> loop.load_state_dict({})
        """

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        r"""Returns a dictionary containing state values.

        Returns
        -------
            dict: The state values in a dict.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.loops.evaluation import EvaluationLoop
            >>> loop = EvaluationLoop()
            >>> state = loop.state_dict()
            >>> state
            {}
        """
