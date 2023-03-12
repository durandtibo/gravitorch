__all__ = ["setup_initializer"]

import logging
from typing import Union

from gravitorch.nn.init.base import BaseInitializer
from gravitorch.nn.init.noop import NoOpInitializer
from gravitorch.utils.format import str_target_object

logger = logging.getLogger(__name__)


def setup_initializer(initializer: Union[BaseInitializer, dict, None]) -> BaseInitializer:
    r"""Sets up the module parameter initializer.

    Args:
    ----
        initializer (``BaseInitializer`` or dict or ``None``):
            Specifies the model parameter initializer or its
            configuration. If ``None``, the ``NoOpInitializer``
            is instantiated.

    Returns:
    -------
        ``BaseInitializer``: The instantiated module parameter
            initializer.
    """
    if initializer is None:
        initializer = NoOpInitializer()
    if isinstance(initializer, dict):
        logger.debug(
            "Initializing a module initializer from its configuration... "
            f"{str_target_object(initializer)}"
        )
        initializer = BaseInitializer.factory(**initializer)
    return initializer
