r"""This module implements some conditions used in the evaluation loops."""

__all__ = ["BaseEvalCondition", "EveryEpochEvalCondition", "LastEpochEvalCondition"]

from abc import abstractmethod

from objectory import AbstractFactory

from gravitorch.engines.base import BaseEngine


class BaseEvalCondition(metaclass=AbstractFactory):
    r"""Defines the base class for the conditions of the evaluation loop.

    These conditions indicate if the evaluation loop should be executed
    or not.
    """

    @abstractmethod
    def __call__(self, engine: BaseEngine) -> bool:
        r"""Indicates if the evaluation loop should be executed.

        Args:
            engine (``BaseEngine``): Specifies the engine.

        Returns:
            bool: ``True`` if the evaluation loop should be evaluated,
                otherwise ``False``
        """


class EveryEpochEvalCondition(BaseEvalCondition):
    r"""Implements a condition that is true every N epoch.

    Args:
        every (int, optional): Specifies the frequency of the
            evaluation. Default: ``1``
    """

    def __init__(self, every: int = 1):
        self._every = every

    def __call__(self, engine: BaseEngine) -> bool:
        r"""Indicates if it is a multiple of ``every``.

        Args:
            engine (``BaseEngine``): Specifies the engine.

        Returns:
            bool: ``True`` if the evaluation loop should be evaluated,
                otherwise ``False``
        """
        return engine.epoch % self._every == 0

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(every={self._every})"


class LastEpochEvalCondition(BaseEvalCondition):
    r"""Implements a condition that is true only for the last epoch."""

    def __call__(self, engine: BaseEngine) -> bool:
        r"""Indicates if it is the last epoch.

        Args:
            engine (``BaseEngine``): Specifies the engine.

        Returns:
            bool: ``True`` if the evaluation loop should be evaluated,
                otherwise ``False``
        """
        return engine.epoch == engine.max_epochs - 1

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"
