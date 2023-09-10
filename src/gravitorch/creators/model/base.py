from __future__ import annotations

__all__ = ["BaseModelCreator", "is_model_creator_config", "setup_model_creator"]

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from objectory import AbstractFactory
from objectory.utils import is_object_config
from torch import nn

from gravitorch.utils.format import str_target_object

if TYPE_CHECKING:
    from gravitorch.engines import BaseEngine

logger = logging.getLogger(__name__)


class BaseModelCreator(ABC, metaclass=AbstractFactory):
    r"""Defines the base class to create a model.

    Note that it is not the unique approach to create a model. Feel
    free to use other approaches if this approach does not fit your
    needs.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.testing import create_dummy_engine
        >>> from gravitorch.creators.model import ModelCreator
        >>> creator = ModelCreator({"_target_": "gravitorch.testing.DummyClassificationModel"})
        >>> creator
        ModelCreator(
          (model_config): {'_target_': 'gravitorch.testing.DummyClassificationModel'}
          (attach_model_to_engine): True
          (add_module_to_engine): True
          (device_placement): AutoDevicePlacement(device=cpu)
        )
        >>> engine = create_dummy_engine()
        >>> model = creator.create(engine)
        >>> model
        DummyClassificationModel(
          (linear): Linear(in_features=4, out_features=3, bias=True)
          (criterion): CrossEntropyLoss()
        )
    """

    @abstractmethod
    def create(self, engine: BaseEngine) -> nn.Module:
        r"""Creates a model on the device(s) where it should run.

        This method is responsible to register the event handlers
        associated to the model. This method is also responsible to
        move the model parameters to the device(s).

        Args:
        ----
            engine (``gravitorch.engines.BaseEngine``): Specifies an
                engine.

        Returns:
        -------
            ``torch.nn.Module``: The created model.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.testing import create_dummy_engine
            >>> from gravitorch.creators.model import ModelCreator
            >>> creator = ModelCreator({"_target_": "gravitorch.testing.DummyClassificationModel"})
            >>> engine = create_dummy_engine()
            >>> model = creator.create(engine)
            >>> model
            DummyClassificationModel(
              (linear): Linear(in_features=4, out_features=3, bias=True)
              (criterion): CrossEntropyLoss()
            )
        """


def is_model_creator_config(config: dict) -> bool:
    r"""Indicates if the input configuration is a configuration for a
    ``BaseModelCreator``.

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
            for a ``BaseModelCreator`` object.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.creators.model import is_model_creator_config
        >>> is_model_creator_config(
        ...     {
        ...         "_target_": "gravitorch.creators.model.ModelCreator",
        ...         "model_config": {"_target_": "gravitorch.testing.DummyClassificationModel"},
        ...     }
        ... )
        True
    """
    return is_object_config(config, BaseModelCreator)


def setup_model_creator(creator: BaseModelCreator | dict) -> BaseModelCreator:
    r"""Sets up the model creator.

    The model creator is instantiated from its configuration by using
    the ``BaseModelCreator`` factory function.

    Args:
    ----
        creator (``BaseModelCreator`` or dict): Specifies the model
            creator or its configuration.

    Returns:
    -------
        ``BaseModelCreator``: The instantiated model creator.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.creators.model import setup_model_creator
        >>> creator = setup_model_creator(
        ...     {
        ...         "_target_": "gravitorch.creators.model.ModelCreator",
        ...         "model_config": {"_target_": "gravitorch.testing.DummyClassificationModel"},
        ...     }
        ... )
        >>> creator
        ModelCreator(
          (model_config): {'_target_': 'gravitorch.testing.DummyClassificationModel'}
          (attach_model_to_engine): True
          (add_module_to_engine): True
          (device_placement): AutoDevicePlacement(device=cpu)
        )
    """
    if isinstance(creator, dict):
        logger.info(
            f"Initializing a model creator from its configuration... {str_target_object(creator)}"
        )
        creator = BaseModelCreator.factory(**creator)
    if not isinstance(creator, BaseModelCreator):
        logger.warning(f"creator is not a `BaseModelCreator` (received: {type(creator)})")
    return creator
