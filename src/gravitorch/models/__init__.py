r"""This package contains the model base class and some implemented
models."""

from __future__ import annotations

__all__ = [
    "BaseModel",
    "VanillaModel",
    "is_model_config",
    "setup_and_attach_model",
    "setup_model",
]

from gravitorch.models.base import BaseModel
from gravitorch.models.utils import is_model_config, setup_and_attach_model, setup_model
from gravitorch.models.vanilla import VanillaModel
