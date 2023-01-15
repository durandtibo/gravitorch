r"""This module defines the base engine state."""

__all__ = ["BaseEngineState"]

from abc import ABC, abstractmethod
from typing import Any, Optional

from objectory import AbstractFactory

from gravitorch.utils.history import BaseHistory


class BaseEngineState(ABC, metaclass=AbstractFactory):
    r"""Defines the base engine state.

    A state should implement the following attributes:

    .. code-block:: python

        >>> from gravitorch.utils.engine_states import BaseEngineState
        >>> state: BaseEngineState = ...  # Create an engine state
        >>> state.epoch             # 0-based, the first epoch is 0. -1 means the training has not started
        >>> state.iteration         # 0-based, the first iteration is 0. -1 means the training has not started
        >>> state.max_epochs        # maximum number of epochs to run
        >>> state.random_seed       # random seed
    """

    @property
    @abstractmethod
    def epoch(self) -> int:
        r"""Gets the epoch value.

        The epoch is 0-based, i.e. the first epoch is 0.
        The value ``-1`` is used to indicate the training has not
        started.

        Returns:
            int: The epoch value.

        Example:

        .. code-block:: python

            >>> from gravitorch.utils.engine_states import BaseEngineState
            >>> state: BaseEngineState = ...  # Create an engine state
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
            int: The iteration value.

        Example:

        .. code-block:: python

            >>> from gravitorch.utils.engine_states import BaseEngineState
            >>> state: BaseEngineState = ...  # Create an engine state
            >>> state.iteration
            -1
        """

    @property
    @abstractmethod
    def max_epochs(self) -> int:
        r"""Gets the maximum number of training epochs.

        Returns:
            int: The maximum number of training epochs.

        Example:

        .. code-block:: python

            >>> from gravitorch.utils.engine_states import BaseEngineState
            >>> state: BaseEngineState = ...  # Create an engine state
            >>> state.max_epochs
            1
        """

    @property
    @abstractmethod
    def random_seed(self) -> int:
        r"""Gets the random seed.

        Returns:
            int: The random seed.

        Example:

        .. code-block:: python

            >>> from gravitorch.utils.engine_states import BaseEngineState
            >>> state: BaseEngineState = ...  # Create an engine state
            >>> state.random_seed
            42
        """

    @abstractmethod
    def add_history(self, history: BaseHistory, key: Optional[str] = None) -> None:
        r"""Adds a history to the state.

        Args:
            history (``BaseHistory``): Specifies the history
                to add to the state.
            key (str or ``None``, optional): Specifies the key to store
                the history. If ``None``, the name of the history is
                used. Default: ``None``

        Example:

        .. code-block:: python

            >>> from gravitorch.utils.engine_states import BaseEngineState
            >>> from gravitorch.utils.history import MinScalarHistory
            >>> state: BaseEngineState = ...  # Create an engine state
            >>> state.add_history(MinScalarHistory("loss"))
            >>> state.add_history(MinScalarHistory("loss"), 'my key')
        """

    @abstractmethod
    def add_module(self, name: str, module: Any) -> None:
        r"""Adds a module to the engine state.

        Note that the name should be unique. If the name exists, the
        old module will be overwritten by the new module.

        Args:
            name (str): Specifies the name of the module to add to
                the engine state.
            module: Specifies the module to add to the enfine state.

        Example usage:

        .. code-block:: python

            >>> from torch import nn
            >>> from gravitorch.utils.engine_states import BaseEngineState
            >>> state: BaseEngineState = ...  # Create an engine state
            >>> state.add_module('model', nn.Linear(4, 6))
        """

    @abstractmethod
    def get_best_values(self, prefix: str = "", suffix: str = "") -> dict[str, Any]:
        r"""Gets the best value of each metric.

        This method ignores the metrics with empty history and the
        non-comparable history.

        Args:
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
            dict: The dict with the best value of each metric.

        Example:

        .. code-block:: python

            >>> from gravitorch.utils.engine_states import BaseEngineState
            # Create an engine state where the best value of 'accuracy' is 42.
            >>> state: BaseEngineState = ...
            >>> best_values = state.get_best_values()
            {'accuracy': 42}
            >>> state.get_best_values(prefix='best/')
            {'best/accuracy': 42}
            >>> state.get_best_values(suffix='/best')
            {'accuracy/best': 42}
        """

    @abstractmethod
    def get_history(self, key: str) -> BaseHistory:
        r"""Gets the history associated to a key.

        Args:
            key (str): Specifies the key of the history to retrieve.

        Returns
            ``BaseHistory``: The history if it exists,
                otherwise it returns an empty history. The created
                empty history is a ``GenericHistory``.

        Example:

        .. code-block:: python

            >>> from gravitorch.utils.engine_states import BaseEngineState
            >>> from gravitorch.utils.history import MinScalarHistory
            >>> state: BaseEngineState = ...  # Create an engine state
            >>> state.add_history(MinScalarHistory("loss"))
            >>> state.get_history('loss')
            MinScalarHistory(name='loss', ...)
            >>> state.get_history('new_history')
            GenericHistory(name='new_history', ...)
        """

    @abstractmethod
    def get_histories(self) -> dict[str, BaseHistory]:
        r"""Gets all histories in the state.

        Returns
            ``dict``: The histories with their keys.

        Example:

        .. code-block:: python

            >>> from gravitorch.utils.engine_states import BaseEngineState
            >>> from gravitorch.utils.history import MinScalarHistory
            >>> state: BaseEngineState = ...  # Create an engine state
            >>> state.add_history(MinScalarHistory("loss"))
            >>> state.get_histories()
            {'loss': MinScalarHistory(name='loss', ...)}
        """

    @abstractmethod
    def get_module(self, name: str) -> Any:
        r"""Gets a module.

        Args:
            name (str): Specifies the module to get.

        Returns:
            The module

        Raises:
            ValueError if the module does not exist.

        Example:

        .. code-block:: python

            >>> from torch import nn
            >>> from gravitorch.utils.engine_states import BaseEngineState
            >>> state: BaseEngineState = ...  # Create an engine state
            >>> state.add_module('model', nn.Linear(4, 6))
            >>> state.get_module('model')
            Linear(in_features=4, out_features=6, bias=True)
            >>> state.get_module('missing_module')
            ValueError
        """

    @abstractmethod
    def has_history(self, key: str) -> bool:
        r"""Indicates if the state has a history for the given key.

        Args:
            key (str): Specifies the key of the history.

        Returns
            bool: ``True`` if the history exists, ``False`` otherwise.

        Example:

        .. code-block:: python

            >>> from gravitorch.utils.engine_states import BaseEngineState
            >>> from gravitorch.utils.history import MinScalarHistory
            >>> state: BaseEngineState = ...  # Create an engine state
            >>> state.add_history(MinScalarHistory("loss"))
            >>> state.has_history('loss')
            True
            >>> state.has_history('missing_history')
            False
        """

    @abstractmethod
    def has_module(self, name: str) -> bool:
        r"""Indicates if there is module for the given name.

        Args:
            name (str): Specifies the name to check.

        Returns:
            bool: ``True`` if the module exists, otherwise ``False``.

        Example:

        .. code-block:: python

            >>> from torch import nn
            >>> from gravitorch.utils.engine_states import BaseEngineState
            >>> state: BaseEngineState = ...  # Create an engine state
            >>> state.add_module('model', nn.Linear(4, 6))
            >>> state.has_module('model')
            True
            >>> state.has_module('missing_module')
            False
        """

    @abstractmethod
    def increment_epoch(self, increment: int = 1) -> None:
        r"""Increments the epoch value by the given value.

        Args:
            increment (int, optional): Specifies the increment for the
                epoch value. Default: ``1``

        Example:

        .. code-block:: python

            >>> from gravitorch.utils.engine_states import BaseEngineState
            >>> state: BaseEngineState = ...  # Create an engine state
            >>> state.epoch
            0
            # Increment the epoch number by 1.
            >>> state.increment_epoch()
            >>> state.epoch
            1
            # Increment the epoch number by 10.
            >>> state.increment_epoch(10)
            >>> state.epoch
            11
        """

    @abstractmethod
    def increment_iteration(self, increment: int = 1) -> None:
        r"""Increments the iteration value by the given value.

        Args:
            increment (int, optional): Specifies the increment for the
                iteration value. Default: ``1``

        Example:

        .. code-block:: python

            >>> from gravitorch.utils.engine_states import BaseEngineState
            >>> state: BaseEngineState = ...  # Create an engine state
            >>> state.iteration
            0
            # Increment the iteration number by 1.
            >>> state.increment_iteration()
            >>> state.iteration
            1
            # Increment the iteration number by 10.
            >>> state.increment_iteration(10)
            >>> state.iteration
            11
        """

    @abstractmethod
    def load_state_dict(self, state_dict: dict) -> None:
        r"""Loads the state values from a dict.

        Args:
            state_dict (dict): A dict with parameters.

        Example:

        .. code-block:: python

            >>> from gravitorch.utils.engine_states import BaseEngineState
            >>> state: BaseEngineState = ...  # Create an engine state
            # Create a state dict. Please look at the implementation of the state_dict method
            # to know the expected structure
            >>> my_state_dict = {...}
            >>> state.load_state_dict(my_state_dict)
        """

    @abstractmethod
    def remove_module(self, name: str) -> None:
        r"""Removes a module from the state.

        Args:
            name (str): Specifies the name of the module to remove.

        Raises:
            ValueError if the module name is not found.

        Example:

        .. code-block:: python

            >>> from torch import nn
            >>> from gravitorch.utils.engine_states import BaseEngineState
            >>> state: BaseEngineState = ...  # Create an engine state
            >>> state.add_module('model', nn.Linear(4, 6))
            >>> state.has_module('model')
            True
            >>> state.remove_module('model')
            >>> state.has_module('model')
            False
            >>> state.remove_module('model')
            ValueError
        """

    @abstractmethod
    def state_dict(self) -> dict:
        r"""Returns a dictionary containing state values.

        Returns:
            dict: the state values in a dict.

        Example:

        .. code-block:: python

            >>> from gravitorch.utils.engine_states import BaseEngineState
            >>> state: BaseEngineState = ...  # Create an engine state
            >>> state.state_dict()
            {...}
        """
