from __future__ import annotations

__all__ = ["is_map_datapipe_config", "setup_map_datapipe"]

import logging

from objectory import factory
from objectory.utils import is_object_config
from torch.utils.data import MapDataPipe

from gravitorch.utils.format import str_target_object

logger = logging.getLogger(__name__)


def is_map_datapipe_config(config: dict) -> bool:
    r"""Indicates if the input configuration is a configuration for a
    ``MapDataPipe``.

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
            for a ``MapDataPipe`` object.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.datapipes import is_map_datapipe_config
        >>> is_map_datapipe_config(
        ...     {
        ...         "_target_": "torch.utils.data.datapipes.map.SequenceWrapper",
        ...         "sequence": [1, 2, 3, 4],
        ...     }
        ... )
        True
    """
    return is_object_config(config, MapDataPipe)


def setup_map_datapipe(datapipe: MapDataPipe | dict) -> MapDataPipe:
    r"""Sets up an ``MapDataPipe``.

    Args:
    ----
        datapipe (``MapDataPipe`` or dict): Specifies a ``MapDataPipe``
            or its configuration.

    Returns:
    -------
        ``MapDataPipe``: The instantiated ``MapDataPipe``.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.datapipes import setup_map_datapipe
        >>> datapipe = setup_map_datapipe(
        ...     {
        ...         "_target_": "torch.utils.data.datapipes.map.SequenceWrapper",
        ...         "sequence": [1, 2, 3, 4],
        ...     }
        ... )
        >>> tuple(datapipe)
        (1, 2, 3, 4)
    """
    if isinstance(datapipe, dict):
        logger.info(
            f"Initializing a `MapDataPipe` from its configuration... {str_target_object(datapipe)}"
        )
        datapipe = factory(**datapipe)
    return datapipe
