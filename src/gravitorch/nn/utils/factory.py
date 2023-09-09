from __future__ import annotations

__all__ = ["attach_module_to_engine", "is_module_config", "setup_module"]

import logging
from typing import TYPE_CHECKING

from objectory import factory
from objectory.utils import is_object_config
from torch.nn import Module

from gravitorch.utils.format import str_target_object

if TYPE_CHECKING:
    from gravitorch.engines import BaseEngine

logger = logging.getLogger(__name__)


def attach_module_to_engine(module: Module, engine: BaseEngine) -> None:
    r"""Attaches a module to the engine if the module has the ``attach``
    method.

    This function does nothing if the module does not have a
    ``attach`` method.

    Args:
    ----
        module (``torch.nn.Module``): Specifies the module to attach
            to the engine.
        engine (``BaseEngine``): Specifies the engine.

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from gravitorch.nn import attach_module_to_engine
        >>> from gravitorch.testing import create_dummy_engine
        >>> engine = create_dummy_engine()
        >>> module = torch.nn.Linear(4, 6)
        >>> attach_module_to_engine(module, engine)
    """
    if hasattr(module, "attach"):
        module.attach(engine)


def is_module_config(config: dict) -> bool:
    r"""Indicates if the input configuration is a configuration for a
    ``torch.nn.Module``.

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
            for a ``torch.nn.Module`` object.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.nn import is_module_config
        >>> is_module_config({"_target_": "torch.nn.Identity"})
        True
    """
    return is_object_config(config, Module)


def setup_module(module: Module | dict) -> Module:
    r"""Sets up a ``torch.nn.Module`` object.

    Args:
    ----
        module (``torch.nn.Module`` or dict): Specifies the module or
            its configuration (dictionary).

    Returns:
    -------
        ``torch.nn.Module``: The instantiated ``torch.nn.Module``
            object.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.nn import setup_module
        >>> linear = setup_module(
        ...     {"_target_": "torch.nn.Linear", "in_features": 4, "out_features": 6}
        ... )
        >>> linear
        Linear(in_features=4, out_features=6, bias=True)
    """
    if isinstance(module, dict):
        logger.info(
            "Initializing a `torch.nn.Module` from its configuration... "
            f"{str_target_object(module)}"
        )
        module = factory(**module)
    if not isinstance(module, Module):
        logger.warning(f"module is not a `torch.nn.Module` (received: {type(module)})")
    return module
