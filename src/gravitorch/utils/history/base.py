r"""This module defines the base class of the history tracker."""

__all__ = [
    "BaseHistory",
    "EmptyHistoryError",
    "HistoryAllCloseOperator",
    "HistoryEqualityOperator",
    "NotAComparableHistoryError",
]

import logging
from abc import ABC, abstractmethod
from typing import Any, Generic, Optional, TypeVar

from coola import (
    AllCloseTester,
    BaseAllCloseOperator,
    BaseAllCloseTester,
    BaseEqualityOperator,
    BaseEqualityTester,
    EqualityTester,
)
from objectory import OBJECT_TARGET, AbstractFactory
from objectory.utils import full_object_name

T = TypeVar("T")

logger = logging.getLogger(__name__)


class BaseHistory(Generic[T], ABC, metaclass=AbstractFactory):
    r"""Definition of a base class to track a history of values.

    The history tracks the value added as well as the step
    when the value is added. The goal of this class is to track the
    recent history because the loggers (e.g. MLFlow or Tensorboard)
    do not allow to get the last value or the best value. The history
    keeps in memory a recent history of pairs (step, value)
    where step is the index of the step when the value was added. The
    length of the recent history depends on the concrete
    implementation.

    To implement your own history, you will need to define the
    following methods:

        - ``add_value``
        - ``get_last_value``
        - ``get_recent_history``
        - ``is_comparable``
        - ``load_state_dict``
        - ``state_dict``

    If it is a comparable history, you will need to implement
    the following methods too:

        - ``_get_best_value``
        - ``_has_improved``

    You may also need to extend the ``config_dict`` method.

    Args:
        name (str): Specifies the name of the history.
    """

    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        r"""str: The name of the history."""
        return self._name

    @abstractmethod
    def add_value(self, value: T, step: Optional[int] = None) -> None:
        r"""Adds a new value to the history.

        Args:
            value: Specifies the value to add to the history.
            step (int or None, optional): Specifies the step value to
                record. Default: ``None``.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from gravitorch.utils.history import GenericHistory
            >>> history = GenericHistory('loss')
            >>> history.add_value(value=2)
            >>> history.add_value(value=torch.zeros(2, 3), step=1)
        """

    def clone(self) -> "BaseHistory":
        r"""Clones the current history.

        Returns:
            ``BaseHistory``: A copy of the current history.

        Example usage:

        .. code-block:: python

            >>> from gravitorch.utils.history import GenericHistory
            >>> history = GenericHistory('epoch')
            >>> history_cloned = history.clone()
        """
        return self.from_dict(self.to_dict())

    @abstractmethod
    def equal(self, other: Any) -> bool:
        r"""Indicates if two histories are equal or not.

        Args:
            other: Specifies the value to compare.

        Returns:
            bool: ``True`` if the histories are equal,
                ``False`` otherwise.

        Example usage:

        .. code-block:: python

            >>> from gravitorch.utils.history import GenericHistory
            >>> history1 = GenericHistory('loss')
            >>> history2 = GenericHistory('accuracy')
            >>> history1.equal(history2)
            False
        """

    def get_best_value(self) -> T:
        r"""Gets the best value of this history.

        It is possible to get the best value only if it is a
        comparable history i.e. it is possible to compare the
        values in the history.

        Returns:
            The best value of this history.

        Raises:
            ``NotAComparableHistory``: if it is not a comparable
                history.
            ``EmptyHistoryError``: if the history is empty

        Example usage:

        .. code-block:: python

            >>> from gravitorch.utils.history import MaxScalarHistory
            >>> history = MaxScalarHistory('accuracy')
            >>> history.add_value(value=2, step=0)
            >>> history.add_value(value=4, step=1)
            >>> history.get_best_value()
            4
        """
        if not self.is_comparable():
            raise NotAComparableHistoryError(
                "It is not possible to get the best value because it is not possible to compare "
                f"the values in {self.name} history"
            )
        return self._get_best_value()

    def _get_best_value(self) -> T:
        r"""Gets the best value of this history.

        You need to implement this method for a comparable history.

        Returns:
            The best value of this history.

        Raises:
            ``NotImplementedError``: if this method is not implemented.
        """
        raise NotImplementedError

    @abstractmethod
    def get_last_value(self) -> T:
        r"""Gets the last value.

        Returns:
            The last value added in the history.

        Example usage:

        .. code-block:: python

            >>> from gravitorch.utils.history import GenericHistory
            >>> history = GenericHistory('loss')
            >>> history.add_value(value=2, step=0)
            >>> history.add_value(value=1.2, step=1)
            >>> history.get_last_value()
            1.2
            >>> history.add_value(value=0.8, step=1)
            >>> history.get_last_value()
            0.8
        """

    @abstractmethod
    def get_recent_history(self) -> tuple[tuple[Optional[int], T], ...]:
        r"""Gets the list of value in the recent history.

        The last value in the tuple is the last value added to the
        history. The length of the recent history depends on the
        concrete implementation.

        Returns:
            tuple: A tuple of the recent history.

        Example usage:

        .. code-block:: python

            >>> from gravitorch.utils.history import GenericHistory
            >>> history = GenericHistory('loss')
            >>> history.add_value(value=2)
            >>> history.add_value(value=1.2, step=1)
            >>> history.add_value(value=0.8, step=2)
            >>> history.get_recent_history()
            ((None, 2), (1, 1.2), (2, 0.8))
        """

    def has_improved(self) -> bool:
        r"""Indicates if the last value is the best value.

        It is possible to use this method only if it is a comparable
        history i.e. it is possible to compare the values in
        the history.

        Returns:
            bool: ``True`` if the last value is the best value,
                otherwise ``False``.

        Raises:
            ``NotAComparableHistory``: if it is not a comparable
                history.
            ``EmptyHistoryError``: if the history is empty

        .. code-block:: python

            >>> from gravitorch.utils.history import MaxScalarHistory
            >>> history = MaxScalarHistory('accuracy')
            >>> history.add_value(value=2, step=0)
            >>> history.add_value(value=4, step=1)
            >>> history.has_improved()
            True
        """
        if not self.is_comparable():
            raise NotAComparableHistoryError(
                "It is not possible to indicate if the last value is the best value because it "
                f"is not possible to compare the values in {self.name} history"
            )
        return self._has_improved()

    def _has_improved(self) -> bool:
        r"""Indicates if the last value is the best value.

        You need to implement this method for a comparable history.

        Returns:
            bool: ``True`` if the last value is the best value,
                otherwise ``False``.

        Raises:
            ``NotImplementedError``: if this method is not implemented
        """
        raise NotImplementedError

    @abstractmethod
    def is_comparable(self) -> bool:
        r"""Indicates if it is possible to compare the values in the history.

        Note that it is possible to compute the best value only for
        histories that are comparable.

        Returns:
            bool: ``True`` if it is possible to compare the values in
            the history, otherwise ``False``.

        .. code-block:: python

            >>> from gravitorch.utils.history import GenericHistory
            >>> history = GenericHistory('loss')
            >>> history.is_comparable()
            False
        """

    @abstractmethod
    def is_empty(self) -> bool:
        r"""Indicates if the history is empty or not.

        Returns:
            bool: ``True`` if the history is empty, otherwise
                ``False``.

        .. code-block:: python

            >>> from gravitorch.utils.history import GenericHistory
            >>> history = GenericHistory('loss')
            >>> history.is_empty()
            True
        """

    def config_dict(self) -> dict[str, Any]:
        r"""Gets the config of the history.

        The config dictionary should contain all the values necessary
        to instantiate a history with the same parameters
        with the  ``factory`` method. It is expected to contain values
        like the full name of the class and the arguments of the
        constructor. This dictionary should not contain the state
        values. It is possible to get the state values with the
        ``state_dict`` method.

        Returns:
            dict: The config of the history

        Example usage:

        .. code-block:: python

            >>> from gravitorch.utils.history import BaseHistory, GenericHistory
            >>> config = GenericHistory('loss').config_dict()
            >>> history = BaseHistory.factory(**config)  # Note that the state is not copied.
        """
        return {
            OBJECT_TARGET: full_object_name(self.__class__),
            "name": self._name,
        }

    @abstractmethod
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        r"""Sets up the history from a dictionary containing the state values.

        Args:
            state_dict (dict): Specifies a dictionary containing state
                keys with values.

        Example usage:

        .. code-block:: python

            >>> from gravitorch.utils.history import GenericHistory
            >>> history = GenericHistory('loss')
            >>> history.load_state_dict({...})
        """

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        r"""Gets a dictionary containing the state values of the history.

        Returns:
            dict: The state values in a dict.

        Example usage:

        .. code-block:: python

            >>> from gravitorch.utils.history import GenericHistory
            >>> state = GenericHistory('loss').state_dict()
        """

    @classmethod
    def from_dict(cls, data: dict) -> "BaseHistory":
        r"""Instantiates a history from a dictionary.

        Args:
            data (dict): Specifies the dictionary that is used to
                instantiate the history. The dictionary is
                expected to contain the parameters to create
                instantiate the history and the state of the
                history.

        Returns:
            ``BaseHistory``: The instantiated history.

        Example usage:

        .. code-block:: python

            >>> from gravitorch.utils.history import BaseHistory
            >>> from objectory import OBJECT_TARGET
            >>> history = BaseHistory.from_dict(
            ...     {
            ...         "config": {
            ...             OBJECT_TARGET: "gravitorch.utils.history.generic.GenericHistory",
            ...             "name": "loss",
            ...             "max_size": 7,
            ...         },
            ...         "state": {"history": ((0, 1), (1, 5))},
            ...     }
            ... )
        """
        obj = cls.factory(**data["config"])
        obj.load_state_dict(data["state"])
        return obj

    def to_dict(self) -> dict[str, Any]:
        r"""Exports the current history to a dictionary.

        This method exports all the information to re-create the
        history with the same state. The returned dictionary
        can be used as input of the ``from_dict`` method to resume the
        history.

        Returns:
            dict: A dictionary with the config and the state of the
                history.

        Example usage:

        .. code-block:: python

            >>> from gravitorch.utils.history import BaseHistory, GenericHistory
            >>> history_dict = GenericHistory('loss').to_dict()
            >>> history = BaseHistory.from_dict(**history_dict)
        """
        return {"config": self.config_dict(), "state": self.state_dict()}


class EmptyHistoryError(Exception):
    r"""Generates an error if the history is empty."""


class NotAComparableHistoryError(Exception):
    r"""Generates an error if it is not possible to compare the values in the
    history."""


class HistoryAllCloseOperator(BaseAllCloseOperator[BaseHistory]):
    r"""Implements an allclose operator for ``BaseHistory`` objects."""

    def allclose(
        self,
        tester: BaseAllCloseTester,
        object1: BaseHistory,
        object2: Any,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        equal_nan: bool = False,
        show_difference: bool = False,
    ) -> bool:
        if not isinstance(object2, BaseHistory):
            if show_difference:
                logger.info(f"object2 is not a `BaseHistory` object: {type(object2)}")
            return False
        object_equal = object1.equal(object2)
        if show_difference and not object_equal:
            logger.info(
                f"`BaseHistory` objects are different\nobject1=\n{object1}\nobject2=\n{object2}"
            )
        return object_equal


class HistoryEqualityOperator(BaseEqualityOperator[BaseHistory]):
    r"""Implements an equality operator for ``BaseHistory`` objects."""

    def equal(
        self,
        tester: BaseEqualityTester,
        object1: BaseHistory,
        object2: Any,
        show_difference: bool = False,
    ) -> bool:
        if not isinstance(object2, BaseHistory):
            if show_difference:
                logger.info(f"object2 is not a `BaseHistory` object: {type(object2)}")
            return False
        object_equal = object1.equal(object2)
        if show_difference and not object_equal:
            logger.info(
                f"`BaseHistory` objects are different\nobject1=\n{object1}\nobject2=\n{object2}"
            )
        return object_equal


if not AllCloseTester.has_allclose_operator(BaseHistory):
    AllCloseTester.add_allclose_operator(BaseHistory, HistoryAllCloseOperator())  # pragma: no cover
if not EqualityTester.has_equality_operator(BaseHistory):
    EqualityTester.add_equality_operator(BaseHistory, HistoryEqualityOperator())  # pragma: no cover
