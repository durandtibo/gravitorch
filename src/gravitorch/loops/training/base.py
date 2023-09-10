r"""This module defines the base class to implement training loops."""

from __future__ import annotations

__all__ = ["BaseTrainingLoop"]

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from objectory import AbstractFactory

if TYPE_CHECKING:
    from gravitorch.engines import BaseEngine

logger = logging.getLogger(__name__)


class BaseTrainingLoop(ABC, metaclass=AbstractFactory):
    r"""Defines the base class ti implement training loops.

    To implement your own training loop, you will need to define the
    following methods:

        - ``train``
        - ``load_state_dict``
        - ``state_dict``

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.loops.training import TrainingLoop
        >>> from gravitorch.testing import create_dummy_engine
        >>> engine = create_dummy_engine()
        >>> loop = TrainingLoop()
        >>> loop
        TrainingLoop(
          (set_grad_to_none): True
          (batch_device_placement): AutoDevicePlacement(device=cpu)
          (tag): train
          (clip_grad_fn): None
          (clip_grad_args): ()
          (observer): NoOpLoopObserver()
          (profiler): NoOpProfiler()
        )
        >>> loop.train(engine)
    """

    @abstractmethod
    def train(self, engine: BaseEngine) -> None:
        r"""Trains the model on the training dataset.

        The training metrics/artifacts should be logged through the
        engine.

        Args:
        ----
            engine (``BaseEngine``): Specifies the engine.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.loops.training import TrainingLoop
            >>> from gravitorch.testing import create_dummy_engine
            >>> engine = create_dummy_engine()
            >>> loop = TrainingLoop()
            >>> loop.train(engine)
        """

    @abstractmethod
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        r"""Sets up the training loop from a dictionary containing the
        state values.

        Args:
        ----
            state_dict (dict): Specifies a dictionary
                containing state keys with values.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.loops.training import TrainingLoop
            >>> loop = TrainingLoop()
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

            >>> from gravitorch.loops.training import TrainingLoop
            >>> loop = TrainingLoop()
            >>> state = loop.state_dict()
            >>> state
            {}
        """
