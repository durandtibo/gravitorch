from __future__ import annotations

__all__ = ["BaseEngine", "is_engine_config", "setup_engine"]

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from minevent import BaseEventHandler
from objectory import AbstractFactory
from objectory.utils import is_object_config
from torch.nn import Module
from torch.optim import Optimizer

from gravitorch.lr_schedulers import LRSchedulerType
from gravitorch.utils.artifacts import BaseArtifact
from gravitorch.utils.exp_trackers.steps import Step
from gravitorch.utils.format import str_target_object
from gravitorch.utils.history import BaseHistory

if TYPE_CHECKING:
    from gravitorch.datasources import BaseDataSource

logger = logging.getLogger(__name__)


class BaseEngine(ABC, metaclass=AbstractFactory):
    r"""Defines the base engine.

    This is an experimental API and the engine design may change in the
    future.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.testing import create_dummy_engine
        >>> engine = create_dummy_engine()
        >>> engine.train()
        >>> engine.eval()
    """

    @property
    @abstractmethod
    def datasource(self) -> BaseDataSource:
        r"""``BaseDataSource``: The datasource object associated to the
        engine.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.testing import create_dummy_engine
            >>> engine = create_dummy_engine()
            >>> engine.datasource
            DummyDataSource(
              (datasets):
                (train): DummyDataset(num_examples=4, feature_size=4)
                (eval): DummyDataset(num_examples=4, feature_size=4)
              (dataloader_creators):
                (train): DataLoaderCreator(
                    (seed): 0
                    (batch_size): 2
                    (shuffle): False
                  )
                (eval): DataLoaderCreator(
                    (seed): 0
                    (batch_size): 2
                    (shuffle): False
                  )
            )
        """

    @property
    @abstractmethod
    def epoch(self) -> int:
        r"""``int``: The epoch value.

        The epoch is 0-based, i.e. the first
        epoch is 0. The value ``-1`` is used to indicate the training
        has not started.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.testing import create_dummy_engine
            >>> engine = create_dummy_engine()
            >>> engine.epoch
            -1
        """

    @property
    @abstractmethod
    def iteration(self) -> int:
        r"""``int``: The iteration value.

        The iteration is 0-based, i.e.
        the first iteration is 0. The value ``-1`` is used to indicate
        the training has not started.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.testing import create_dummy_engine
            >>> engine = create_dummy_engine()
            >>> engine.iteration
            -1
        """

    @property
    @abstractmethod
    def lr_scheduler(self) -> LRSchedulerType | None:
        r"""``LRSchedulerType`` or ``None``: The learning rate (LR)
        scheduler if it is defined, otherwise ``None``.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.testing import create_dummy_engine
            >>> engine = create_dummy_engine()
            >>> engine.lr_scheduler
            None
        """

    @property
    @abstractmethod
    def max_epochs(self) -> int:
        r"""``int``: The maximum number of training epochs.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.testing import create_dummy_engine
            >>> engine = create_dummy_engine()
            >>> engine.max_epochs
            1
        """

    @property
    @abstractmethod
    def model(self) -> Module:
        r"""``torch.nn.Module``: The model to train and/or evaluate.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.testing import create_dummy_engine
            >>> engine = create_dummy_engine()
            >>> engine.model
            DummyClassificationModel(
              (linear): Linear(in_features=4, out_features=3, bias=True)
              (criterion): CrossEntropyLoss()
            )
        """

    @property
    @abstractmethod
    def optimizer(self) -> Optimizer | None:
        r"""``torch.nn.Optimizer`` or ``None``: The optimizer to train
        the model.

        It can be ``None`` if the model is not trained.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.testing import create_dummy_engine
            >>> engine = create_dummy_engine()
            >>> engine.optimizer
            SGD (
            Parameter Group 0
                dampening: 0
                differentiable: False
                foreach: None
                lr: 0.01
                maximize: False
                momentum: 0
                nesterov: False
                weight_decay: 0
            )
        """

    @property
    @abstractmethod
    def random_seed(self) -> int:
        r"""``int``: The random seed.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.testing import create_dummy_engine
            >>> engine = create_dummy_engine()
            >>> engine.random_seed
            9984043075503325450
        """

    @property
    @abstractmethod
    def should_terminate(self) -> bool:
        r"""``bool``: Flag to indicate if this engine should terminate
        training at the end of the current epoch.

        If ``True``, the engine should terminate at the end of the
        current epoch. Use the ``terminate()`` method to set this flag
        to ``True``.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.testing import create_dummy_engine
            >>> engine = create_dummy_engine()
            >>> engine.should_terminate
            False
        """

    @abstractmethod
    def add_event_handler(self, event: str, event_handler: BaseEventHandler) -> None:
        r"""Adds an event handler to an event.

        The event handler will be called everytime the event happens.

        Args:
        ----
            event (str): Specifies the event to attach the event
                handler.
            event_handler (``BaseEventHandler``): Specifies the event
                handler to attach to the event.

        Example usage:

        .. code-block:: pycon

            >>> from minevent import EventHandler
            >>> from gravitorch.testing import create_dummy_engine
            >>> engine = create_dummy_engine()
            >>> def hello_handler():
            ...     print("Hello!")
            ...
            >>> engine.add_event_handler("my_event", EventHandler(hello_handler))
        """

    @abstractmethod
    def add_history(self, history: BaseHistory, key: str | None = None) -> None:
        r"""Adds a history to the engine state.

        Args:
        ----
            history (``BaseHistory``): Specifies the history to
                add to the engine state.
            key (str or ``None``, optional): Specifies the key to
                store the history. If ``None``, the name of the
                history is used. Default: ``None``

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.history import MinScalarHistory
            >>> from gravitorch.testing import create_dummy_engine
            >>> engine = create_dummy_engine()
            >>> engine.add_history(MinScalarHistory("loss"))
            >>> engine.add_history(MinScalarHistory("loss"), "my key")
        """

    @abstractmethod
    def add_module(self, name: str, module: Any) -> None:
        r"""Adds a module to the engine.

        The state dict of the module will be added to the engine
        state when a checkpoint is created.

        Args:
        ----
            name (str): Specifies the name of the module.
            module: Specifies the module to add to the checkpoint. The
                module needs to have the ``state_dict`` and
                ``load_state_dict`` methods if you want to store the
                state of the module in the engine checkpoint. If the
                module does not have ``state_dict`` and
                ``load_state_dict`` methods, it will be ignored when
                the state dict is created or loaded.

        Example usage:

        .. code-block:: pycon

            >>> from torch import nn
            >>> from gravitorch.testing import create_dummy_engine
            >>> engine = create_dummy_engine()
            >>> engine.add_module("my_module", nn.Linear(4, 5))
        """

    @abstractmethod
    def create_artifact(self, artifact: BaseArtifact) -> None:
        r"""Creates an artifact.

        Args:
        ----
            artifact (``BaseArtifact``): Specifies the artifact to
            create.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.artifacts import JSONArtifact
            >>> from gravitorch.testing import create_dummy_engine
            >>> engine = create_dummy_engine()
            >>> engine.create_artifact(JSONArtifact(tag="metric", data={"f1_score": 42}))
        """

    @abstractmethod
    def eval(self) -> None:
        r"""Evaluates the model on the given evaluation dataset with the
        given metrics/loss.

        The evaluation dataset has to be defined.

        Raises
        ------
            RuntimeError: if the evaluation dataset is not be defined.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.testing import create_dummy_engine
            >>> engine = create_dummy_engine()
            >>> engine.eval()
        """

    @abstractmethod
    def trigger_event(self, event: str) -> None:
        r"""Triggers the handler(s) for the given event.

        Args:
        ----
            event (str): Specifies the event to trigger.

        Example usage:

        .. code-block:: pycon

            >>> from minevent import EventHandler
            >>> from gravitorch.testing import create_dummy_engine
            >>> engine = create_dummy_engine()
            >>> engine.trigger_event("my_event")  # should do nothing because there is no event handler
            >>> def hello_handler():
            ...     print("Hello!")
            ...
            >>> engine.add_event_handler("my_event", EventHandler(hello_handler))
            >>> engine.trigger_event("my_event")
            Hello!
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

        Example:
        -------
        .. code-block:: pycon

            >>> from gravitorch.utils.history import MinScalarHistory
            >>> from gravitorch.testing import create_dummy_engine
            >>> engine = create_dummy_engine()
            >>> engine.add_history(MinScalarHistory("loss"))
            >>> engine.get_history("loss")
            MinScalarHistory(name=loss, max_size=10, history=())
            >>> engine.get_history("new_history")
            GenericHistory(name=new_history, max_size=10, history=())
        """

    @abstractmethod
    def get_histories(self) -> dict[str, BaseHistory]:
        r"""Gets all histories in the state.

        Returns:
        -------
            ``dict``: The histories with their keys.

        Example:
        -------
        .. code-block:: pycon

            >>> from gravitorch.utils.history import MinScalarHistory
            >>> from gravitorch.testing import create_dummy_engine
            >>> engine = create_dummy_engine()
            >>> engine.add_history(MinScalarHistory("loss"))
            >>> engine.get_histories()
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

        Example:
        -------
        .. code-block:: pycon

            >>> from torch import nn
            >>> from gravitorch.testing import create_dummy_engine
            >>> engine = create_dummy_engine()
            >>> engine.add_module("model", nn.Linear(4, 6))
            >>> engine.get_module("model")
            Linear(in_features=4, out_features=6, bias=True)
        """

    @abstractmethod
    def has_event_handler(self, event_handler: BaseEventHandler, event: str | None = None) -> bool:
        r"""Indicates if a handler is registered in the event manager.

        Note that this method relies on the ``__eq__`` method of the
        given event handler to compare event handlers.

        Args:
        ----
            event_handler (``BaseEventHandler``): Specifies the event
                handler to check.
            event (str or ``None``): Specifies an event to check. If
                the value is ``None``, it will check all the events.
                Default: ``None``

        Example usage:

        .. code-block:: pycon

            >>> from minevent import EventHandler
            >>> from gravitorch.testing import create_dummy_engine
            >>> engine = create_dummy_engine()
            >>> def hello_handler():
            ...     print("Hello!")
            ...
            >>> engine.has_event_handler(EventHandler(hello_handler))
            False
            >>> engine.has_event_handler(EventHandler(hello_handler), "my_event")
            False
            >>> engine.add_event_handler("my_event", EventHandler(hello_handler))
            >>> engine.has_event_handler(EventHandler(hello_handler))
            True
            >>> engine.has_event_handler(EventHandler(hello_handler), "my_event")
            True
            >>> engine.has_event_handler(EventHandler(hello_handler), "other_event")
            False
        """

    @abstractmethod
    def has_history(self, key: str) -> bool:
        r"""Indicates if the engine has a history for the given key.

        Args:
        ----
            key (str): Specifies the key of the history.

        Returns:
        -------
            bool: ``True`` if the history exists, ``False`` otherwise.

        Example:
        -------
        .. code-block:: pycon

            >>> from gravitorch.utils.history import MinScalarHistory
            >>> from gravitorch.testing import create_dummy_engine
            >>> engine = create_dummy_engine()
            >>> engine.add_history(MinScalarHistory("loss"))
            >>> engine.has_history("loss")
            True
            >>> engine.has_history("missing_history")
            False
        """

    @abstractmethod
    def has_module(self, name: str) -> bool:
        r"""Indicates if the engine has a module for the given name.

        Args:
        ----
            name (str): Specifies the name of the module.

        Returns:
        -------
            bool: ``True`` if the module exists, otherwise ``False``.

        Example:
        -------
        .. code-block:: pycon

            >>> from torch import nn
            >>> from gravitorch.testing import create_dummy_engine
            >>> engine = create_dummy_engine()
            >>> engine.add_module("model", nn.Linear(4, 6))
            >>> engine.has_module("model")
            True
            >>> engine.has_module("missing_module")
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

            >>> from gravitorch.testing import create_dummy_engine
            >>> engine = create_dummy_engine()
            >>> engine.epoch
            -1
            >>> # Increment the epoch number by 1.
            >>> engine.increment_epoch()
            >>> engine.epoch
            0
            >>> # Increment the epoch number by 10.
            >>> engine.increment_epoch(10)
            >>> engine.epoch
            10
        """

    @abstractmethod
    def increment_iteration(self, increment: int = 1) -> None:
        r"""Increments the iteration value by the given value.

        Args:
        ----
            increment (int, optional): Specifies the increment for
                the iteration value. Default: ``1``

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.testing import create_dummy_engine
            >>> engine = create_dummy_engine()
            >>> engine.iteration
            -1
            >>> # Increment the iteration number by 1.
            >>> engine.increment_iteration()
            >>> engine.iteration
            0
            >>> # Increment the iteration number by 10.
            >>> engine.increment_iteration(10)
            >>> engine.iteration
            10
        """

    @abstractmethod
    def load_state_dict(self, state_dict: dict) -> None:
        r"""Loads the state values from a dict.

        Args:
        ----
            state_dict (dict): a dict with parameters

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from gravitorch.testing import create_dummy_engine
            >>> engine = create_dummy_engine()
            >>> engine.load_state_dict(
            ...     {
            ...         "epoch": -1,
            ...         "iteration": -1,
            ...         "histories": {},
            ...         "modules": {
            ...             "training_loop": {},
            ...             "evaluation_loop": {},
            ...             "datasource": {},
            ...             "model": {
            ...                 "linear.weight": torch.ones(3, 4),
            ...                 "linear.bias": torch.ones(3),
            ...             },
            ...             "optimizer": {
            ...                 "state": {},
            ...                 "param_groups": [
            ...                     {
            ...                         "lr": 0.01,
            ...                         "momentum": 0,
            ...                         "dampening": 0,
            ...                         "weight_decay": 0,
            ...                         "nesterov": False,
            ...                         "maximize": False,
            ...                         "foreach": None,
            ...                         "differentiable": False,
            ...                         "params": [0, 1],
            ...                     }
            ...                 ],
            ...             },
            ...         },
            ...     }
            ... )
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
        ----
            key (str): Specifies the key used to identify the figure.
            figure (``matplotlib.pyplot.Figure``): Specifies the
                figure to log.
            step (``Step``, optional): Specifies the step value to
                record. If ``None``, it will use the default step.
                Default: ``None``

        Example usage:

        .. code-block:: pycon

            >>> import matplotlib.pyplot as plt
            >>> from gravitorch.utils.exp_trackers import EpochStep, IterationStep
            >>> from gravitorch.testing import create_dummy_engine
            >>> engine = create_dummy_engine()
            >>> fig, ax = plt.subplots()
            >>> engine.log_figure("my_figure", fig)
            >>> engine.log_figure("my_figure", fig, EpochStep(2))
            >>> engine.log_figure("my_figure", fig, IterationStep(10))
        """

    @abstractmethod
    def log_figures(
        self,
        figures: dict[str, matplotlib.pyplot.Figure],  # noqa: F821
        step: Step | None = None,
    ) -> None:
        r"""Logs a dictionary of figures for a given step.

        Args:
        ----
            figures (dict): Specifies the dictionary of figures to log.
            step (``Step``, optional): Specifies the step value to
                record. If ``None``, it will use the default step.
                Default: ``None``

        Example usage:

        .. code-block:: pycon

            >>> import matplotlib.pyplot as plt
            >>> from gravitorch.utils.exp_trackers import EpochStep, IterationStep
            >>> from gravitorch.testing import create_dummy_engine
            >>> engine = create_dummy_engine()
            >>> fig, ax = plt.subplots()
            >>> engine.log_figures({"my_figure_1": fig, "my_figure_2": fig})
            >>> engine.log_figures({"my_figure_1": fig, "my_figure_2": fig}, EpochStep(2))
            >>> engine.log_figures({"my_figure_1": fig, "my_figure_2": fig}, IterationStep(10))
        """

    @abstractmethod
    def log_metric(self, key: str, value: int | float, step: Step | None = None) -> None:
        r"""Logs a single metric.

        It is possible to have access to the last logged values by
        using the ``get_history(key)`` method.

        Args:
        ----
            key (str): Specifies the key used to identify the scalar
                value.
            value (int or float): Specifies the value to log.
            step (``Step``, optional): Specifies the step value to
                record. If ``None``, it will use the default step.
                Default: ``None``

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.exp_trackers import EpochStep, IterationStep
            >>> from gravitorch.testing import create_dummy_engine
            >>> engine = create_dummy_engine()
            >>> engine.log_metric(key="metric name", value=1.2)
            >>> engine.log_metric(key="metric name", value=1.2, step=EpochStep(2))
            >>> engine.log_metric(key="metric name", value=1.2, step=IterationStep(10))
        """

    @abstractmethod
    def log_metrics(self, metrics: dict[str, int | float], step: Step | None = None) -> None:
        r"""Logs a dictionary of multiple metrics.

        It is possible to have access to the last logged values by
        using the ``get_history(key)`` method.

        Args:
        ----
            metrics (dict): Specifies the dictionary of metric to log.
            step (``Step``, optional): Specifies the step value to
                record. If ``None``, it will use the default step.
                Default: ``None``

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.exp_trackers import EpochStep, IterationStep
            >>> from gravitorch.testing import create_dummy_engine
            >>> engine = create_dummy_engine()
            >>> metrics = {"metric1": 12, "metric2": 0.35}
            >>> engine.log_metrics(metrics)
            >>> engine.log_metrics(metrics, EpochStep(2))
            >>> engine.log_metrics(metrics, IterationStep(10))
        """

    @abstractmethod
    def remove_event_handler(self, event: str, event_handler: BaseEventHandler) -> None:
        r"""Removes an event handler of a given event.

        Note that if the same event handler was added multiple times
        the event, all the duplicated handlers are removed. This
        method relies on the ``__eq__`` method of the input event
        handler to compare event handlers.

        Args:
        ----
            event (str): Specifies the event handler is attached to.
            event_handler (``BaseEventHandler``): Specifies the event
                handler to remove.

        Raises:
        ------
            ValueError: if the event does not exist or if the handler
                is not attached to the event.

        Example usage:

        .. code-block:: pycon

            >>> from minevent import EventHandler
            >>> from gravitorch.testing import create_dummy_engine
            >>> engine = create_dummy_engine()
            >>> def hello_handler():
            ...     print("Hello!")
            ...
            >>> engine.add_event_handler("my_event", EventHandler(hello_handler))
            >>> engine.has_event_handler(EventHandler(hello_handler), "my_event")
            True
            >>> engine.remove_event_handler("my_event", EventHandler(hello_handler))
            >>> engine.has_event_handler(EventHandler(hello_handler), "my_event")
            False
        """

    @abstractmethod
    def remove_module(self, name: str) -> None:
        r"""Removes a module from the engine state.

        Args:
        ----
            name (str): Specifies the name of the module to remove.

        Raises:
        ------
            ValueError if the module name is not found.

        Example:
        -------
        .. code-block:: pycon

            >>> from torch import nn
            >>> from gravitorch.testing import create_dummy_engine
            >>> engine = create_dummy_engine()
            >>> engine.add_module("model", nn.Linear(4, 6))
            >>> engine.has_module("model")
            True
            >>> engine.remove_module("model")
            >>> engine.has_module("model")
            False
        """

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        r"""Gets a dictionary containing a state of the engine.

        Note this state does not capture the handler states.

        Returns
        -------
            dict: The dictionary containing a state of the engine.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.testing import create_dummy_engine
            >>> engine = create_dummy_engine()
            >>> state = engine.state_dict()
            >>> state
            {'epoch': -1,
             'iteration': -1,
             'histories': {},
             'modules': {'training_loop': {}, 'evaluation_loop': {}, 'datasource': {}, 'model': OrderedDict([...]), 'optimizer': {...}}}
        """

    @abstractmethod
    def terminate(self) -> None:
        r"""Sends terminate signal to the engine to stop the training
        after the current epoch.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.testing import create_dummy_engine
            >>> engine = create_dummy_engine()
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

        .. code-block:: pycon

            >>> from gravitorch.testing import create_dummy_engine
            >>> engine = create_dummy_engine()
            >>> engine.train()
        """


def is_engine_config(config: dict) -> bool:
    r"""Indicates if the input configuration is a configuration for a
    ``BaseEngine``.

    This function only checks if the value of the key  ``_target_``
    is valid. It does not check the other values. If ``_target_``
    indicates a function, the returned type hint is used to check
    the class.

    Args:
    ----
        config (dict): Specifies the configuration to check.

    Returns:
    -------
        bool: ``True`` if the input configuration is a configuration
            for a ``BaseEngine`` object.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.engines import is_engine_config
        >>> is_engine_config({"_target_": "gravitorch.engines.AlphaEngine"})
        True
    """
    return is_object_config(config, BaseEngine)


def setup_engine(engine: BaseEngine | dict) -> BaseEngine:
    r"""Sets up an engine.

    The engine is instantiated from its configuration by using
    the ``BaseEngine`` factory function.

    Args:
    ----
        engine (``BaseEngine`` or dict): Specifies the data
            source or its configuration.

    Returns:
    -------
        ``BaseEngine``: The instantiated engine.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.engines import setup_engine
        >>> engine = setup_engine(
        ...     {
        ...         "_target_": "gravitorch.engines.AlphaEngine",
        ...         "core_creator": {
        ...             "_target_": "gravitorch.creators.core.CoreCreator",
        ...             "datasource": {"_target_": "gravitorch.testing.DummyDataSource"},
        ...             "model": {"_target_": "gravitorch.testing.DummyClassificationModel"},
        ...         },
        ...     }
        ... )
        >>> engine
        AlphaEngine(...)
    """
    if isinstance(engine, dict):
        logger.info(f"Initializing a engine from its configuration... {str_target_object(engine)}")
        engine = BaseEngine.factory(**engine)
    if not isinstance(engine, BaseEngine):
        logger.warning(f"engine is not a BaseEngine (received: {type(engine)})")
    return engine
