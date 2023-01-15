r"""This module defines the base model."""

__all__ = ["attach_module_to_engine", "setup_model", "setup_and_attach_model"]

import logging
from typing import Union

from torch.nn import Module

from gravitorch.engines.base import BaseEngine
from gravitorch.models.base import BaseModel
from gravitorch.utils.format import str_target_object

logger = logging.getLogger(__name__)


def attach_module_to_engine(module: Module, engine: BaseEngine) -> None:
    r"""Attaches a module to the engine if the module has the ``attach`` method.

    This function does nothing if the module does not have a
    ``attach`` method.

    Args:
        module (``torch.nn.Module``): Specifies the module to attach
            to the engine.
        engine (``BaseEngine``): Specifies the engine.
    """
    if hasattr(module, "attach"):
        module.attach(engine)


def setup_model(model: Union[Module, dict]) -> Module:
    r"""Sets up the model.

    The model is instantiated from its configuration by using the
    ``BaseModel`` factory function.

    Args:
        model (``torch.nn.Module`` or dict): Specifies the model or
            its configuration.

    Returns:
        ``torch.nn.Module``: The (instantiated) model.
    """
    if isinstance(model, dict):
        logger.info(f"Initializing a model from its configuration... {str_target_object(model)}")
        model = BaseModel.factory(**model)
    return model


def setup_and_attach_model(engine: BaseEngine, model: Union[Module, dict]) -> Module:
    r"""Sets up and attaches the model to the engine.

    The model is attached to the engine by using the ``attach`` method.
     If the model does not have a ``attach`` method, the ``attach``
     step is skipped.

    Note that if you call this function ``N`` times with the same
    model, the model will be attached ``N`` times to the engine.

    Args:
        engine (``BaseEngine``): Specifies the engine.
        model (``torch.nn.Module`` or dict): Specifies the model or
            its configuration.

    Returns:
        ``torch.nn.Module``: The (instantiated) model.
    """
    model = setup_model(model)
    if hasattr(model, "attach"):
        logger.info("Adding a model to the engine state...")
        model.attach(engine=engine)
    return model
