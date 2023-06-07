from __future__ import annotations

__all__ = ["setup_object"]

import logging
from typing import TypeVar

from objectory import factory

from gravitorch.utils.format import str_target_object

logger = logging.getLogger(__name__)

T = TypeVar("T")


def setup_object(obj_or_config: T | dict) -> T:
    r"""Sets up an object from its configuration.

    Args:
    ----
        obj_or_config: Specifies the object or its configuration.

    Returns:
    -------
        The instantiated object.

    Example usage:

    .. code-block:: pycon

       >>> from gravitorch.utils import setup_object
       >>> linear = setup_object(
       ...     {"_target_": "torch.nn.Linear", "in_features": 4, "out_features": 6}
       ... )
       >>> linear
       Linear(in_features=4, out_features=6, bias=True)
       >>> setup_object(linear)  # Do nothing because the module is already instantiated
       Linear(in_features=4, out_features=6, bias=True)
    """
    if isinstance(obj_or_config, dict):
        logger.info(
            "Initializing an object from its configuration... "
            f"{str_target_object(obj_or_config)}"
        )
        obj_or_config = factory(**obj_or_config)
    return obj_or_config
