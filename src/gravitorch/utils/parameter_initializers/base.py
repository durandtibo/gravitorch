r"""This module defines the base class for the model parameter
parameter_initializers."""

__all__ = ["BaseParameterInitializer", "BaseDefaultParameterInitializer"]

import logging
from abc import ABC, abstractmethod

from objectory import AbstractFactory

from gravitorch.engines.base import BaseEngine
from gravitorch.nn.utils.parameter_analysis import show_parameter_stats

logger = logging.getLogger(__name__)


class BaseParameterInitializer(ABC, metaclass=AbstractFactory):
    r"""Defines the base parameter initializer.

    Note that there are other ways to initialize the model parameters.
    For example, you can initialize the weights of your model directly
    in the model.
    """

    @abstractmethod
    def initialize(self, engine: BaseEngine) -> None:
        r"""Initializes the parameters of the model.

        Note that the input is the engine instead of the model because
        it allows to initialize the weights based on the dataset or
        other parameters in the engine. The parameters of the model
        should be updated in-place.

        Args:
            engine (``BaseEngine``): Specifies the engine.
        """


class BaseDefaultParameterInitializer(BaseParameterInitializer):
    r"""Defines a base parameter initializer that shows some stats about the
    parameters.

    Note that there are other ways to initialize the model parameters.
    For example, you can initialize the weights of your model directly
    in the model.

    Args:
        show_stats (bool, optional): If ``True``, the parameter
            statistics are shown at the end of the initialization.
            Default: ``True``
    """

    def __init__(self, show_stats: bool = True):
        self._show_stats = show_stats

    def initialize(self, engine: BaseEngine) -> None:
        r"""Initializes the parameters of the model and shows some stats.

        Note that the input is the engine instead of the model because
        it allows to initialize the weights based on the dataset or
        other parameters in the engine. The parameters of the model
        should be updated in-place.

        Args:
            engine (``BaseEngine``): Specifies the engine.
        """
        self._initialize(engine)
        if self._show_stats:
            show_parameter_stats(engine.model)

    @abstractmethod
    def _initialize(self, engine: BaseEngine) -> None:
        r"""Initializes the parameters of the model.

        Note that the input is the engine instead of the model because
        it allows to initialize the weights based on the dataset or
        other parameters in the engine. The parameters of the model
        should be updated in-place.

        Args:
            engine (``BaseEngine``): Specifies the engine.
        """
