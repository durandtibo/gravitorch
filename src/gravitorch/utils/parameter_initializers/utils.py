r"""This module implements some utility functions for the parameter
initializers."""

__all__ = ["setup_parameter_initializer"]

import logging
from typing import Union

from gravitorch.utils.format import str_target_object
from gravitorch.utils.parameter_initializers.base import BaseParameterInitializer
from gravitorch.utils.parameter_initializers.no_init import NoParameterInitializer

logger = logging.getLogger(__name__)


def setup_parameter_initializer(
    parameter_initializer: Union[BaseParameterInitializer, dict, None]
) -> BaseParameterInitializer:
    r"""Sets up the model parameter initializer.

    Args:
        parameter_initializer (``BaseParameterInitializer`` or dict or
            ``None``): Specifies the model parameter initializer or
            its configuration. If ``None``, the
            ``NoParameterInitializer`` will be instantiated.

    Returns:
        ``BaseParameterInitializer``: The instantiated model parameter
            initializer.
    """
    if parameter_initializer is None:
        parameter_initializer = NoParameterInitializer()
    if isinstance(parameter_initializer, dict):
        logger.debug(
            "Initializing a parameter initializer from its configuration... "
            f"{str_target_object(parameter_initializer)}"
        )
        parameter_initializer = BaseParameterInitializer.factory(**parameter_initializer)
    return parameter_initializer
