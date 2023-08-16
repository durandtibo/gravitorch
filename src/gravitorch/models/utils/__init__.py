r"""This package contains some utility functions for the models."""

from __future__ import annotations

__all__ = [
    "ModelSummary",
    "analyze_model_architecture",
    "analyze_module_architecture",
    "analyze_network_architecture",
    "attach_module_to_engine",
    "is_loss_decreasing",
    "is_loss_decreasing_with_adam",
    "is_loss_decreasing_with_sgd",
    "is_model_config",
    "setup_and_attach_model",
    "setup_model",
]

from gravitorch.models.utils.architecture import (
    analyze_model_architecture,
    analyze_module_architecture,
    analyze_network_architecture,
)
from gravitorch.models.utils.factory import (
    attach_module_to_engine,
    is_model_config,
    setup_and_attach_model,
    setup_model,
)
from gravitorch.models.utils.summary import ModelSummary
from gravitorch.models.utils.testing import (
    is_loss_decreasing,
    is_loss_decreasing_with_adam,
    is_loss_decreasing_with_sgd,
)
