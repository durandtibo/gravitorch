__all__ = ["setup_module"]

import logging
from typing import Union

from objectory import factory
from torch.nn import Module

from gravitorch.utils.format import str_target_object

logger = logging.getLogger(__name__)


def setup_module(module: Union[Module, dict]) -> Module:
    r"""Sets up a ``torch.nn.Module`` object.

    Args:
        module (``torch.nn.Module`` or dict): Specifies the module or
            its configuration (dictionary).

    Returns:
        ``torch.nn.Module``: The instantiated ``torch.nn.Module``
            object.

    Example usage:

    .. code-block:: python

        >>> from gravitorch.nn import setup_module
        >>> linear = setup_module(
        ...     {'_target_': 'torch.nn.Linear', 'in_features': 4, 'out_features': 6}
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
    return module
