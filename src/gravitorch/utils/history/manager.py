r"""This file implements a history manager."""

from __future__ import annotations

__all__ = ["HistoryManager"]

import copy
import logging
from typing import Any

from coola.utils import str_indent, str_mapping

from gravitorch.utils.history.base import BaseHistory
from gravitorch.utils.history.generic import GenericHistory
from gravitorch.utils.history.utils import get_best_values

logger = logging.getLogger(__name__)


class HistoryManager:
    r"""Implements a manager to easily manage a group of histories.

    This class proposes an approach to manage a group of histories, but
    it is possible to use other approaches. If this class does not fit
    your needs, feel free to use another approach.
    """

    def __init__(self) -> None:
        self._histories = dict()

    def __len__(self) -> int:
        r"""``int``: The number of histories in the manager."""
        return len(self._histories)

    def __repr__(self) -> str:
        if self._histories:
            return (
                f"{self.__class__.__qualname__}(\n"
                f"  {str_indent(str_mapping(self._histories))}\n)"
            )
        return f"{self.__class__.__qualname__}()"

    def add_history(self, history: BaseHistory, key: str | None = None) -> None:
        r"""Adds a history to the manager.

        Args:
        ----
            history (``BaseHistory``): Specifies the history
                to add to the manager.
            key (str or ``None``, optional): Specifies the key to
                store the history. If ``None``, the name of the
                history is used. Default: ``None``

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.history import HistoryManager, MinScalarHistory
            >>> manager = HistoryManager()
            >>> manager.add_history(MinScalarHistory("loss"))
            >>> manager
            HistoryManager(
              (loss): MinScalarHistory(name=loss, max_size=10, history=())
            )
            >>> manager.add_history(MinScalarHistory("loss"), "my key")
            >>> manager
            HistoryManager(
              (loss): MinScalarHistory(name=loss, max_size=10, history=())
              (my key): MinScalarHistory(name=loss, max_size=10, history=())
            )
        """
        if key is None:
            key = history.name
        if key in self._histories:
            logger.warning(
                f"The {key} history already exists and will be replace by the new history"
            )
        self._histories[key] = history

    def get_best_values(self, prefix: str = "", suffix: str = "") -> dict[str, Any]:
        """Gets the best value of each metric.

        This method ignores the metrics with empty history and the
        non-comparable history.

        Args:
        ----
            prefix (str): Specifies the prefix used to create the dict
                of best values. The goal of this prefix is to generate
                a name which is different from the metric name to
                avoid confusion. By default, the returned dict uses the
                same name as the metric. Default: ``''``
            suffix (str): Specifies the suffix used to create the dict
                of best values. The goal of this suffix is to generate
                a name which is different from the metric name to
                avoid confusion. By default, the returned dict uses the
                same name as the metric. Default: ``''``

        Returns:
        -------
            dict: The dict with the best value of each metric.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.history import HistoryManager, MaxScalarHistory
            >>> manager = HistoryManager()
            >>> manager.add_history(MaxScalarHistory("accuracy"))
            >>> manager.get_history("accuracy").add_value(42.0)
            >>> manager.get_best_values()
            {'accuracy': 42.0}
            >>> manager.get_best_values(prefix="best/")
            {'best/accuracy': 42.0}
            >>> manager.get_best_values(suffix="/best")
            {'accuracy/best': 42.0}
        """
        return get_best_values(self._histories, prefix=prefix, suffix=suffix)

    def get_history(self, key: str) -> BaseHistory:
        r"""Gets the history associated to a key.

        Args:
        ----
            key (str): Specifies the key of the history to retrieve.

        Returns:
        -------
            ``BaseHistory``: The history if it exists,
                otherwise it returns an empty history. The created
                empty history is a ``GenericHistory``.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.history import HistoryManager, MinScalarHistory
            >>> manager = HistoryManager()
            >>> manager.add_history(MinScalarHistory("loss"))
            >>> manager.get_history("loss")
            MinScalarHistory(name=loss, max_size=10, history=())
            >>> manager.get_history("new_history")
            GenericHistory(name=new_history, max_size=10, history=())
        """
        if not self.has_history(key):
            self._histories[key] = GenericHistory(name=key)
        return self._histories[key]

    def get_histories(self) -> dict[str, BaseHistory]:
        r"""Gets all the histories.

        Returns:
        -------
            ``dict``: The histories with their keys.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.history import HistoryManager, MinScalarHistory
            >>> manager = HistoryManager()
            >>> manager.add_history(MinScalarHistory("loss"))
            >>> manager.get_histories()
            {'loss': MinScalarHistory(name=loss, max_size=10, history=())}
        """
        return copy.copy(self._histories)

    def has_history(self, key: str) -> bool:
        r"""Indicates if the engine has a history for the given key.

        Args:
        ----
            key (str): Specifies the key of the history.

        Returns:
        -------
            bool: ``True`` if the history exists, ``False`` otherwise

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.history import HistoryManager, MinScalarHistory
            >>> manager = HistoryManager()
            >>> manager.add_history(MinScalarHistory("loss"))
            >>> manager.has_history("loss")
            True
            >>> manager.has_history("missing")
            False
        """
        return key in self._histories

    def load_state_dict(self, state_dict: dict) -> None:
        r"""Loads the state values from a dict.

        Args:
        ----
            state_dict (dict): a dict with parameters

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.history import HistoryManager, GenericHistory
            >>> manager = HistoryManager()
            >>> manager.add_history(GenericHistory("value"))
            >>> manager.load_state_dict(
            ...     {"value": {"state": {"history": ((0, 1), (1, 0.5), (2, 0.25))}}}
            ... )
            >>> manager.get_history("value").get_last_value()
            0.25
        """
        for key, state in state_dict.items():
            if self.has_history(key):
                self._histories[key].load_state_dict(state["state"])
            else:
                self._histories[key] = BaseHistory.from_dict(state)

    def state_dict(self) -> dict:
        r"""Returns a dictionary containing state values of all the
        histories.

        Returns:
        -------
            dict: the state values in a dict.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.history import HistoryManager
            >>> manager = HistoryManager()
            >>> manager.state_dict()
            {}
        """
        return {key: hist.to_dict() for key, hist in self._histories.items()}
