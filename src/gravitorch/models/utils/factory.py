r"""This module defines the base model."""

from __future__ import annotations

__all__ = ["is_model_config", "setup_model", "setup_and_attach_model"]

import logging
from typing import TYPE_CHECKING

from objectory.utils import is_object_config
from torch.nn import Module

from gravitorch.models.base import BaseModel
from gravitorch.utils.format import str_target_object

if TYPE_CHECKING:
    from gravitorch.engines import BaseEngine


logger = logging.getLogger(__name__)


def is_model_config(config: dict) -> bool:
    r"""Indicates if the input configuration is a configuration for a
    ``BaseModel``.

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
            for a ``BaseModel`` object.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.models import is_model_config
        >>> is_model_config({"_target_": "gravitorch.models.VanillaModel"})
        True
    """
    return is_object_config(config, BaseModel)


def setup_model(model: Module | dict) -> Module:
    r"""Sets up the model.

    The model is instantiated from its configuration by using the
    ``BaseModel`` factory function.

    Args:
    ----
        model (``torch.nn.Module`` or dict): Specifies the model or
            its configuration.

    Returns:
    -------
        ``torch.nn.Module``: The (instantiated) model.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.models import is_model_config
        >>> model = setup_model(
        ...     {
        ...         "_target_": "gravitorch.models.VanillaModel",
        ...         "network": {"_target_": "torch.nn.Linear", "in_features": 4, "out_features": 6},
        ...         "criterion": None,
        ...     }
        ... )
        >>> model
        VanillaModel(
          (network): Linear(in_features=4, out_features=6, bias=True)
          (metrics): ModuleDict()
        )
    """
    if isinstance(model, dict):
        logger.info(f"Initializing a model from its configuration... {str_target_object(model)}")
        model = BaseModel.factory(**model)
    return model


def setup_and_attach_model(engine: BaseEngine, model: Module | dict) -> Module:
    r"""Sets up and attaches the model to the engine.

    The model is attached to the engine by using the ``attach`` method.
     If the model does not have a ``attach`` method, the ``attach``
     step is skipped.

    Note that if you call this function ``N`` times with the same
    model, the model will be attached ``N`` times to the engine.

    Args:
    ----
        engine (``BaseEngine``): Specifies the engine.
        model (``torch.nn.Module`` or dict): Specifies the model or
            its configuration.

    Returns:
    -------
        ``torch.nn.Module``: The (instantiated) model.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.models import is_model_config
        >>> from gravitorch.testing import create_dummy_engine
        >>> engine = create_dummy_engine()
        >>> model = setup_and_attach_model(
        ...     engine=engine,
        ...     model={
        ...         "_target_": "gravitorch.models.VanillaModel",
        ...         "network": {"_target_": "torch.nn.Linear", "in_features": 4, "out_features": 6},
        ...         "criterion": None,
        ...     },
        ... )
        >>> model
        VanillaModel(
          (network): Linear(in_features=4, out_features=6, bias=True)
          (metrics): ModuleDict()
        )
    """
    model = setup_model(model)
    if hasattr(model, "attach"):
        logger.info("Adding a model to the engine state...")
        model.attach(engine=engine)
    return model
