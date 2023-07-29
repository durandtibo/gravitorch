from __future__ import annotations

__all__ = ["setup_module"]

import logging

from objectory import factory
from objectory.utils import is_object_config
from torch.nn import Module

from gravitorch.utils.format import str_target_object

logger = logging.getLogger(__name__)


def is_module_config(config: dict) -> bool:
    r"""Indicate if the input configuration is a configuration for a
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
        >>> setup_module(linear)  # Do nothing because the module is already instantiated
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
