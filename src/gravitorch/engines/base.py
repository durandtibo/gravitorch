from __future__ import annotations

__all__ = ["BaseEngine"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from objectory import AbstractFactory
from torch.nn import Module
from torch.optim import Optimizer

from gravitorch.lr_schedulers import LRSchedulerType
from gravitorch.utils.artifacts import BaseArtifact
from gravitorch.utils.events import BaseEventHandler
from gravitorch.utils.exp_trackers.steps import Step
from gravitorch.utils.history import BaseHistory

if TYPE_CHECKING:
    from gravitorch.datasources import BaseDataSource


class BaseEngine(ABC, metaclass=AbstractFactory):
    r"""Defines the base engine.

    This is an experimental API and the engine design may change in the
    future.
    """

    @property
    @abstractmethod
    def data_source(self) -> BaseDataSource:
        r"""``BaseDataSource``: The data source object associated to
        the engine."""

    @property
    @abstractmethod
    def epoch(self) -> int:
        r"""int: The epoch value. The epoch is 0-based, i.e. the first
        epoch is 0. The value ``-1`` is used to indicate the training
        has not started.
        """

    @property
    @abstractmethod
    def iteration(self) -> int:
        r"""int: The iteration value. The iteration is 0-based, i.e.
        the first iteration is 0. The value ``-1`` is used to indicate
        the training has not started.
        """

    @property
    @abstractmethod
    def lr_scheduler(self) -> LRSchedulerType | None:
        r"""``LRSchedulerType`` or ``None``: The learning rate (LR) scheduler
        if it is defined, otherwise ``None``."""

    @property
    @abstractmethod
    def max_epochs(self) -> int:
        r"""int: The maximum number of training epochs."""

    @property
    @abstractmethod
    def model(self) -> Module:
        r"""``torch.nn.Module``: The model to train and/or evaluate."""

    @property
    @abstractmethod
    def optimizer(self) -> Optimizer | None:
        r"""``torch.nn.Optimizer`` or ``None``: The optimizer to train the
        model.

        It can be ``None`` if the model is not trained.
        """

    @property
    @abstractmethod
    def random_seed(self) -> int:
        r"""int: The random seed."""

    @property
    @abstractmethod
    def should_terminate(self) -> bool:
        r"""bool: Flag to indicate if this engine should terminate training
        at the end of the current epoch.

        If ``True``, the engine should terminate at the end of the
        current epoch. Use the ``terminate()`` method to set this flag
        to ``True``.
        """

    @abstractmethod
    def add_event_handler(self, event: str, event_handler: BaseEventHandler) -> None:
        r"""Adds an event handler to an event.

        The event handler will be called everytime the event happens.

        Args:
            event (str): Specifies the event to attach the event
                handler.
            event_handler (``BaseEventHandler``): Specifies the event
                handler to attach to the event.

        Example usage:

        .. code-block:: python

            >>> from gravitorch.engines import BaseEngine
            >>> from gravitorch.utils.events import VanillaEventHandler
            >>> engine: BaseEngine = ...  # Create an engine
            >>> def hello_handler():
            ...     print('Hello!')
            >>> engine.add_event_handler('my_event', VanillaEventHandler(hello_handler))
        """

    @abstractmethod
    def add_history(self, history: BaseHistory, key: str | None = None) -> None:
        r"""Adds a history to the engine state.

        Args:
            history (``BaseHistory``): Specifies the history to
                add to the engine state.
            key (str or ``None``, optional): Specifies the key to
                store the history. If ``None``, the name of the
                history is used. Default: ``None``

        Example usage:

        .. code-block:: python

            >>> from gravitorch.engines import BaseEngine
            >>> from gravitorch.utils.history import MinScalarHistory
            >>> engine: BaseEngine = ...  # Create an engine
            >>> engine.add_history(MinScalarHistory("loss"))
            >>> engine.add_history(MinScalarHistory("loss"), 'my key')
        """

    @abstractmethod
    def add_module(self, name: str, module: Any) -> None:
        r"""Adds a module to the engine.

        The state dict of the module will be added to the engine
        state when a checkpoint is created.

        Args:
            name (str): Specifies the name of the module.
            module: Specifies the module to add to the checkpoint. The
                module needs to have the ``state_dict`` and
                ``load_state_dict`` methods if you want to store the
                state of the module in the engine checkpoint. If the
                module does not have ``state_dict`` and
                ``load_state_dict`` methods, it will be ignored when
                the state dict is created or loaded.

        Example usage:

        .. code-block:: python

            >>> from torch import nn
            >>> from gravitorch.engines import BaseEngine
            >>> engine: BaseEngine = ...  # Create an engine
            >>> engine.add_module('my_module', nn.Linear(4, 5))
        """

    @abstractmethod
    def create_artifact(self, artifact: BaseArtifact) -> None:
        r"""Creates an artifact.

        Args:
            artifact (``BaseArtifact``): Specifies the artifact to
            create.

        Example usage:

        .. code-block:: python

            >>> from gravitorch.engines import BaseEngine
            >>> from gravitorch.utils.artifacts import JSONArtifact
            >>> engine: BaseEngine = ...  # Create an engine
            >>> engine.create_artifact(JSONArtifact(tag='metric', data={'f1_score': 42}))
        """

    @abstractmethod
    def eval(self) -> None:
        r"""Evaluates the model on the given evaluation dataset with the given
        metrics/loss.

        The evaluation dataset has to be defined.

        Raises:
            RuntimeError: if the evaluation dataset is not be defined.

        Example usage:

        .. code-block:: python

            >>> from gravitorch.engines import BaseEngine
            >>> engine: BaseEngine = ...  # Create an engine
            >>> engine.eval()
        """

    @abstractmethod
    def fire_event(self, event: str) -> None:
        r"""Fires the handler(s) for the given event.

        Args:
            event (str): Specifies the event to fire.

        Example usage:

        .. code-block:: python

            >>> from gravitorch.engines import BaseEngine
            >>> from gravitorch.utils.events import VanillaEventHandler
            >>> engine: BaseEngine = ...  # Create an engine
            >>> engine.fire_event('my_event')  # should do nothing because there is no event handler
            >>> def hello_handler():
            ...     print("Hello!")
            >>> engine.add_event_handler('my_event', VanillaEventHandler(hello_handler))
            >>> engine.fire_event('my_event')
            Hello!
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

            >>> from gravitorch.engines import BaseEngine
            >>> from gravitorch.utils.history import MinScalarHistory
            >>> engine: BaseEngine = ...  # Create an engine
            >>> engine.add_history(MinScalarHistory("loss"))
            >>> engine.get_history('loss')
            MinScalarHistory(name='loss', ...)
            >>> engine.get_history('new_history')
            GenericHistory(name='new_history', ...)
        """

    @abstractmethod
    def get_histories(self) -> dict[str, BaseHistory]:
        r"""Gets all histories in the state.

        Returns
            ``dict``: The histories with their keys.

        Example:

        .. code-block:: python

            >>> from gravitorch.engines import BaseEngine
            >>> from gravitorch.utils.history import MinScalarHistory
            >>> engine: BaseEngine = ...  # Create an engine
            >>> engine.add_history(MinScalarHistory("loss"))
            >>> engine.get_histories()
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
            >>> from gravitorch.engines import BaseEngine
            >>> engine: BaseEngine = ...  # Create an engine
            >>> engine.add_module('model', nn.Linear(4, 6))
            >>> engine.get_module('model')
            Linear(in_features=4, out_features=6, bias=True)
            >>> engine.get_module('missing_module')
            ValueError
        """

    @abstractmethod
    def has_event_handler(self, event_handler: BaseEventHandler, event: str | None = None) -> bool:
        r"""Indicates if a handler is registered in the event manager.

        Note that this method relies on the ``__eq__`` method of the
        given event handler to compare event handlers.

        Args:
            event_handler (``BaseEventHandler``): Specifies the event
                handler to check.
            event (str or ``None``): Specifies an event to check. If
                the value is ``None``, it will check all the events.
                Default: ``None``

        Example usage:

        .. code-block:: python

            >>> from gravitorch.engines import BaseEngine
            >>> from gravitorch.utils.events import VanillaEventHandler
            >>> engine: BaseEngine = ...  # Create an engine
            # Define a handler
            >>> def hello_handler():
            ...     print("Hello!")
            >>> engine.has_event_handler(VanillaEventHandler(hello_handler))
            False
            >>> engine.has_event_handler(VanillaEventHandler(hello_handler), 'my_event')
            False
            >>> engine.add_event_handler('my_event', VanillaEventHandler(hello_handler))
            >>> engine.has_event_handler(VanillaEventHandler(hello_handler))
            True
            >>> engine.has_event_handler(VanillaEventHandler(hello_handler), 'my_event')
            True
            >>> engine.has_event_handler(VanillaEventHandler(hello_handler), 'other_event')
            False
        """

    @abstractmethod
    def has_history(self, key: str) -> bool:
        r"""Indicates if the engine has a history for the given key.

        Args:
            key (str): Specifies the key of the history.

        Returns
            bool: ``True`` if the history exists, ``False`` otherwise.

        Example:

        .. code-block:: python

            >>> from gravitorch.engines import BaseEngine
            >>> from gravitorch.utils.history import MinScalarHistory
            >>> engine: BaseEngine = ...  # Create an engine
            >>> engine.add_history(MinScalarHistory("loss"))
            >>> engine.has_history('loss')
            True
            >>> engine.has_history('missing_history')
            False
        """

    @abstractmethod
    def has_module(self, name: str) -> bool:
        r"""Indicates if the engine has a module for the given name.

        Args:
            name (str): Specifies the name of the module.

        Returns:
            bool: ``True`` if the module exists, otherwise ``False``.

        Example:

        .. code-block:: python

            >>> from torch import nn
            >>> from gravitorch.engines import BaseEngine
            >>> engine: BaseEngine = ...  # Create an engine
            >>> engine.add_module('model', nn.Linear(4, 6))
            >>> engine.has_module('model')
            True
            >>> engine.has_module('missing_module')
            False
        """

    @abstractmethod
    def increment_epoch(self, increment: int = 1) -> None:
        r"""Increments the epoch value by the given value.

        Args:
            increment (int, optional): Specifies the increment for the
                epoch value. Default: ``1``

        Example usage:

        .. code-block:: python

            >>> from gravitorch.engines import BaseEngine
            >>> engine: BaseEngine = ...  # Create an engine
            >>> engine.epoch
            1
            # Increment the epoch number by 1.
            >>> engine.increment_epoch()
            >>> engine.epoch
            0
            # Increment the epoch number by 10.
            >>> engine.increment_epoch(10)
            >>> engine.epoch
            10
        """

    @abstractmethod
    def increment_iteration(self, increment: int = 1) -> None:
        r"""Increments the iteration value by the given value.

        Args:
            increment (int, optional): Specifies the increment for
                the iteration value. Default: ``1``

        Example usage:

        .. code-block:: python

            >>> from gravitorch.engines import BaseEngine
            >>> engine: BaseEngine = ...  # Create an engine
            >>> engine.iteration
            -1
            # Increment the iteration number by 1.
            >>> engine.increment_iteration()
            >>> engine.iteration
            0
            # Increment the iteration number by 10.
            >>> engine.increment_iteration(10)
            >>> engine.iteration
            10
        """

    @abstractmethod
    def load_state_dict(self, state_dict: dict) -> None:
        r"""Loads the state values from a dict.

        Args:
            state_dict (dict): a dict with parameters

        Example usage:

        .. code-block:: python

            >>> from gravitorch.engines import BaseEngine
            >>> engine: BaseEngine = ...  # Create an engine
            >>> state = {...}  # The content of the state dict depends on the engine.
            >>> engine.load_state_dict(state)
        """

    @abstractmethod
    def log_figure(
        self,
        key: str,
        figure: matplotlib.pyplot.Figure,  # noqa: F821
        step: Step | None = None,
    ) -> None:
        r"""Logs a figure for the given key and step.

        Args:
            key (str): Specifies the key used to identify the figure.
            figure (``matplotlib.pyplot.Figure``): Specifies the
                figure to log.
            step (``Step``, optional): Specifies the step value to
                record. If ``None``, it will use the default step.
                Default: ``None``

        Example usage:

        .. code-block:: python

            >>> import matplotlib.pyplot as plt
            >>> from gravitorch.engines import BaseEngine
            >>> from gravitorch.utils.exp_trackers import EpochStep, IterationStep
            >>> engine: BaseEngine = ...  # Create an engine
            >>> fig, axes = plt.subplots()
            >>> engine.log_figure('my_figure', fig)
            >>> engine.log_figure('my_figure', fig, EpochStep(2))
            >>> engine.log_figure('my_figure', fig, IterationStep(10))
        """

    @abstractmethod
    def log_figures(
        self,
        figures: dict[str, matplotlib.pyplot.Figure],  # noqa: F821
        step: Step | None = None,
    ) -> None:
        r"""Logs a dictionary of figures for a given step.

        Args:
            figures (dict): Specifies the dictionary of figures to log.
            step (``Step``, optional): Specifies the step value to
                record. If ``None``, it will use the default step.
                Default: ``None``

        Example usage:

        .. code-block:: python

            >>> import matplotlib.pyplot as plt
            >>> from gravitorch.engines import BaseEngine
            >>> from gravitorch.utils.exp_trackers import EpochStep, IterationStep
            >>> engine: BaseEngine = ...  # Create an engine
            >>> fig, axes = plt.subplots()
            >>> engine.log_figures({'my_figure_1': fig, 'my_figure_2': fig})
            >>> engine.log_figures({'my_figure_1': fig, 'my_figure_2': fig}, EpochStep(2))
            >>> engine.log_figures({'my_figure_1': fig, 'my_figure_2': fig}, IterationStep(10))
        """

    @abstractmethod
    def log_metric(self, key: str, value: int | float, step: Step | None = None):
        r"""Logs a single metric.

        It is possible to have access to the last logged values by
        using the ``get_history(key)`` method.

        Args:
            key (str): Specifies the key used to identify the scalar
                value.
            value (int or float): Specifies the value to log.
            step (``Step``, optional): Specifies the step value to
                record. If ``None``, it will use the default step.
                Default: ``None``

        Example usage:

        .. code-block:: python

            >>> from gravitorch.engines import BaseEngine
            >>> from gravitorch.utils.exp_trackers import EpochStep, IterationStep
            >>> engine: BaseEngine = ...  # Create an engine
            >>> engine.log_metric(key='metric name', value=1.2)
            >>> engine.log_metric(key='metric name', value=1.2, step=EpochStep(2))
            >>> engine.log_metric(key='metric name', value=1.2, step=IterationStep(10))
        """

    @abstractmethod
    def log_metrics(self, metrics: dict[str, int | float], step: Step | None = None) -> None:
        r"""Logs a dictionary of multiple metrics.

        It is possible to have access to the last logged values by
        using the ``get_history(key)`` method.

        Args:
            metrics (dict): Specifies the dictionary of metric to log.
            step (``Step``, optional): Specifies the step value to
                record. If ``None``, it will use the default step.
                Default: ``None``

        Example usage:

        .. code-block:: python

            >>> from gravitorch.engines import BaseEngine
            >>> from gravitorch.utils.exp_trackers import EpochStep, IterationStep
            >>> engine: BaseEngine = ...  # Create an engine
            >>> my_metrics = {'metric1': 12, 'metric2': 0.35}
            >>> engine.log_metrics(metrics)
            >>> engine.log_metrics(my_metrics, EpochStep(2))
            >>> engine.log_metrics(my_metrics, IterationStep(10))
        """

    @abstractmethod
    def remove_event_handler(self, event: str, event_handler: BaseEventHandler) -> None:
        r"""Removes an event handler of a given event.

        Note that if the same event handler was added multiple times
        the event, all the duplicated handlers are removed. This
        method relies on the ``__eq__`` method of the input event
        handler to compare event handlers.

        Args:
            event (str): Specifies the event handler is attached to.
            event_handler (``BaseEventHandler``): Specifies the event
                handler to remove.

        Raises:
            ValueError: if the event does not exist or if the handler
                is not attached to the event.

        Example usage:

        .. code-block:: python

            >>> from gravitorch.engines import BaseEngine
            >>> from gravitorch.utils.events import VanillaEventHandler
            >>> engine: BaseEngine = ...  # Create an engine
            >>> def hello_handler():
            ...     print('Hello!')
            >>> engine.add_event_handler('my_event', VanillaEventHandler(hello_handler))
            >>> engine.has_event_handler(VanillaEventHandler(hello_handler), 'my_event')
            True
            >>> engine.remove_event_handler('my_event', VanillaEventHandler(hello_handler))
            >>> engine.has_event_handler(VanillaEventHandler(hello_handler), 'my_event')
            False
        """

    @abstractmethod
    def remove_module(self, name: str) -> None:
        r"""Removes a module from the engine state.

        Args:
            name (str): Specifies the name of the module to remove.

        Raises:
            ValueError if the module name is not found.

        Example:

        .. code-block:: python

            >>> from torch import nn
            >>> from gravitorch.engines import BaseEngine
            >>> engine: BaseEngine = ...  # Create an engine
            >>> engine.add_module('model', nn.Linear(4, 6))
            >>> engine.has_module('model')
            True
            >>> engine.remove_module('model')
            >>> engine.has_module('model')
            False
            >>> engine.remove_module('missing_model')
            ValueError
        """

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        r"""Gets a dictionary containing a state of the engine.

        Note this state does not capture the handler states.

        Returns:
            dict: The dictionary containing a state of the engine.

        Example usage:

        .. code-block:: python

            >>> from gravitorch.engines import BaseEngine
            >>> engine: BaseEngine = ...  # Create an engine
            >>> state = engine.state_dict()
        """

    @abstractmethod
    def terminate(self) -> None:
        r"""Sends terminate signal to the engine to stop the training after the
        current epoch.

        Example usage:

        .. code-block:: python

            >>> from gravitorch.engines import BaseEngine
            >>> engine: BaseEngine = ...  # Create an engine
            >>> engine.should_terminate
            False
            >>> engine.terminate()
            >>> engine.should_terminate
            True
        """

    @abstractmethod
    def train(self) -> None:
        r"""Trains the model on the given training dataset.

        If an evaluation dataset is given, the model will be evaluated
        on the evaluation dataset at each epoch.

        Example usage:

        .. code-block:: python

            >>> from gravitorch.engines import BaseEngine
            >>> engine: BaseEngine = ...  # Create an engine
            >>> engine.train()
        """
