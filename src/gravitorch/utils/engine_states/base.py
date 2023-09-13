r"""This module defines the base engine state."""

from __future__ import annotations

__all__ = ["BaseEngineState"]

from abc import ABC, abstractmethod
from typing import Any

from objectory import AbstractFactory

from gravitorch.utils.history import BaseHistory


class BaseEngineState(ABC, metaclass=AbstractFactory):
    r"""Defines the base engine state.

    A state should implement the following attributes:

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.utils.engine_states import EngineState
        >>> state = EngineState()
        >>> state
        EngineState(
          (modules): AssetManager(num_assets=0)
          (histories): HistoryManager()
          (random_seed): 9984043075503325450
          (max_epochs): 1
          (epoch): -1
          (iteration): -1
        )
        >>> state.epoch  # 0-based, the first epoch is 0. -1 means the training has not started
        >>> state.iteration  # 0-based, the first iteration is 0. -1 means the training has not started
        >>> state.max_epochs  # maximum number of epochs to run
        >>> state.random_seed  # random seed
    """

    @property
    @abstractmethod
    def epoch(self) -> int:
        r"""Gets the epoch value.

        The epoch is 0-based, i.e. the first epoch is 0.
        The value ``-1`` is used to indicate the training has not
        started.

        Returns:
        -------
            int: The epoch value.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.engine_states import EngineState
            >>> state = EngineState()
            >>> state.epoch
            -1
        """

    @property
    @abstractmethod
    def iteration(self) -> int:
        r"""Gets the iteration value.

        The iteration is 0-based, i.e. the first iteration is 0.
        The value ``-1`` is used to indicate the training has not
        started.

        Returns:
        -------
            int: The iteration value.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.engine_states import EngineState
            >>> state = EngineState()
            >>> state.iteration
            -1
        """

    @property
    @abstractmethod
    def max_epochs(self) -> int:
        r"""Gets the maximum number of training epochs.

        Returns:
        -------
            int: The maximum number of training epochs.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.engine_states import EngineState
            >>> state = EngineState()
            >>> state.max_epochs
            1
        """

    @property
    @abstractmethod
    def random_seed(self) -> int:
        r"""Gets the random seed.

        Returns:
        -------
            int: The random seed.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.engine_states import EngineState
            >>> state = EngineState(random_seed=42)
            >>> state.random_seed
            42
        """

    @abstractmethod
    def add_history(self, history: BaseHistory, key: str | None = None) -> None:
        r"""Adds a history to the state.

         Args:
         ----
             history (``BaseHistory``): Specifies the history
                 to add to the state.
             key (str or ``None``, optional): Specifies the key to store
                 the history. If ``None``, the name of the history is
                 used. Default: ``None``

        Example usage:

         .. code-block:: pycon

             >>> from gravitorch.utils.history import MinScalarHistory
             >>> from gravitorch.utils.engine_states import EngineState
             >>> state = EngineState()
             >>> state.add_history(MinScalarHistory("loss"))
             >>> state.add_history(MinScalarHistory("loss"), "my key")
        """

    @abstractmethod
    def add_module(self, name: str, module: Any) -> None:
        r"""Adds a module to the engine state.

        Note that the name should be unique. If the name exists, the
        old module will be overwritten by the new module.

        Args:
        ----
            name (str): Specifies the name of the module to add to
                the engine state.
            module: Specifies the module to add to the enfine state.

        Example usage:

        .. code-block:: pycon

            >>> from torch import nn
            >>> from gravitorch.utils.engine_states import EngineState
            >>> state = EngineState()
            >>> state.add_module("model", nn.Linear(4, 6))
        """

    @abstractmethod
    def get_best_values(self, prefix: str = "", suffix: str = "") -> dict[str, Any]:
        r"""Gets the best value of each metric.

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
                a name which is different from the metric name to avoid
                confusion. By default, the returned dict uses the same
                name as the metric. Default: ``''``

        Returns:
        -------
            dict: The dict with the best value of each metric.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.engine_states import EngineState
            >>> state = EngineState()
            >>> from gravitorch.utils.history import MaxScalarHistory
            >>> history = MaxScalarHistory("accuracy")
            >>> history.add_value(23.0)
            >>> history.add_value(42.0)
            >>> state.add_history(history)
            >>> state.get_best_values()
            {'accuracy': 42.0}
            >>> state.get_best_values(prefix="best/")
            {'best/accuracy': 42.0}
            >>> state.get_best_values(suffix="/best")
            {'accuracy/best': 42.0}
        """

    @abstractmethod
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

            >>> from gravitorch.utils.engine_states import EngineState
            >>> state = EngineState()
            >>> from gravitorch.utils.history import MinScalarHistory
            >>> state.add_history(MinScalarHistory("loss"))
            >>> state.get_history("loss")
            MinScalarHistory(name=loss, max_size=10, history=())
            >>> state.get_history("new_history")
            GenericHistory(name=new_history, max_size=10, history=())
        """

    @abstractmethod
    def get_histories(self) -> dict[str, BaseHistory]:
        r"""Gets all histories in the state.

        Returns:
        -------
            ``dict``: The histories with their keys.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.engine_states import EngineState
            >>> state = EngineState()
            >>> from gravitorch.utils.history import MinScalarHistory
            >>> state.add_history(MinScalarHistory("loss"))
            >>> state.get_histories()
            {'loss': MinScalarHistory(name=loss, max_size=10, history=())}
        """

    @abstractmethod
    def get_module(self, name: str) -> Any:
        r"""Gets a module.

        Args:
        ----
            name (str): Specifies the module to get.

        Returns:
        -------
            The module

        Raises:
        ------
            ValueError if the module does not exist.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.engine_states import EngineState
            >>> state = EngineState()
            >>> from torch import nn
            >>> state.add_module("model", nn.Linear(4, 6))
            >>> state.get_module("model")
            Linear(in_features=4, out_features=6, bias=True)
        """

    @abstractmethod
    def has_history(self, key: str) -> bool:
        r"""Indicates if the state has a history for the given key.

        Args:
        ----
            key (str): Specifies the key of the history.

        Returns:
        -------
            bool: ``True`` if the history exists, ``False`` otherwise.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.engine_states import EngineState
            >>> state = EngineState()
            >>> from gravitorch.utils.history import MinScalarHistory
            >>> state.add_history(MinScalarHistory("loss"))
            >>> state.has_history("loss")
            True
            >>> state.has_history("missing_history")
            False
        """

    @abstractmethod
    def has_module(self, name: str) -> bool:
        r"""Indicates if there is module for the given name.

        Args:
        ----
            name (str): Specifies the name to check.

        Returns:
        -------
            bool: ``True`` if the module exists, otherwise ``False``.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.engine_states import EngineState
            >>> state = EngineState()
            >>> from torch import nn
            >>> state.add_module("model", nn.Linear(4, 6))
            >>> state.has_module("model")
            True
            >>> state.has_module("missing_module")
            False
        """

    @abstractmethod
    def increment_epoch(self, increment: int = 1) -> None:
        r"""Increments the epoch value by the given value.

        Args:
        ----
            increment (int, optional): Specifies the increment for the
                epoch value. Default: ``1``

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.engine_states import EngineState
            >>> state = EngineState()
            >>> state.epoch
            -1
            >>> # Increment the epoch number by 1.
            >>> state.increment_epoch()
            >>> state.epoch
            0
            >>> # Increment the epoch number by 10.
            >>> state.increment_epoch(10)
            >>> state.epoch
            10
        """

    @abstractmethod
    def increment_iteration(self, increment: int = 1) -> None:
        r"""Increments the iteration value by the given value.

         Args:
         ----
             increment (int, optional): Specifies the increment for the
                 iteration value. Default: ``1``

        Example usage:

         .. code-block:: pycon

             >>> from gravitorch.utils.engine_states import EngineState
             >>> state = EngineState()
             >>> state.iteration
             -1
             >>> # Increment the iteration number by 1.
             >>> state.increment_iteration()
             >>> state.iteration
             0
             >>> # Increment the iteration number by 10.
             >>> state.increment_iteration(10)
             >>> state.iteration
             10
        """

    @abstractmethod
    def load_state_dict(self, state_dict: dict) -> None:
        r"""Loads the state values from a dict.

        Args:
        ----
            state_dict (dict): A dict with parameters.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.engine_states import EngineState
            >>> state = EngineState()
            >>> state.load_state_dict({"epoch": 4, "iteration": 42, "histories": {}, "modules": {}})
            >>> state.epoch
            4
            >>> state.iteration
            42
        """

    @abstractmethod
    def remove_module(self, name: str) -> None:
        r"""Removes a module from the state.

        Args:
        ----
            name (str): Specifies the name of the module to remove.

        Raises:
        ------
            ValueError if the module name is not found.

        Example usage:

        .. code-block:: pycon

            >>> from torch import nn
            >>> from gravitorch.utils.engine_states import EngineState
            >>> state = EngineState()
            >>> state.add_module("model", nn.Linear(4, 6))
            >>> state.has_module("model")
            True
            >>> state.remove_module("model")
            >>> state.has_module("model")
            False
        """

    @abstractmethod
    def state_dict(self) -> dict:
        r"""Returns a dictionary containing state values.

        Returns:
        -------
            dict: the state values in a dict.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.engine_states import EngineState
            >>> state = EngineState()
            >>> state.state_dict()
            {'epoch': -1, 'iteration': -1, 'histories': {}, 'modules': {}}
        """
