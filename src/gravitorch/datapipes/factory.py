from __future__ import annotations

__all__ = ["create_chained_datapipe", "is_datapipe_config", "setup_datapipe"]

import logging
from collections.abc import Sequence

from objectory import OBJECT_TARGET, factory
from torch.utils.data import IterDataPipe, MapDataPipe

from gravitorch.datapipes.iter import is_iter_datapipe_config
from gravitorch.datapipes.map import is_map_datapipe_config
from gravitorch.utils.format import str_target_object

logger = logging.getLogger(__name__)


def create_chained_datapipe(configs: Sequence[dict]) -> IterDataPipe | MapDataPipe:
    r"""Creates a chain of ``DataPipe``s from their configuration.

    Args:
    ----
        configs: Specifies the config of each ``DataPipe`` in the
            sequence. The configs sequence follows the order of the
            ``DataPipe``s. The first config is used to create the
            first ``DataPipe`` (a.k.a. source), and the last
            config is used to create the last ``DataPipe`` (a.k.a.
            sink). This function assumes all the ``DataPipes``
            have a single source DataPipe as their first argument,
            excepts for the first DataPipe.

    Returns:
    -------
        ``IterDataPipe`` or ``MapDataPipe``: The last ``DataPipe``
            of the chain (a.k.a. sink).

    Raises:
    ------
        RuntimeError if the ``DataPipe`` configuration sequence is
            empty.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.datapipes import create_chained_datapipe
        >>> datapipe = create_chained_datapipe(
        ...     [
        ...         {
        ...             "_target_": "torch.utils.data.datapipes.iter.IterableWrapper",
        ...             "iterable": [1, 2, 3, 4],
        ...         },
        ...     ],
        ... )
        >>> tuple(datapipe)
        (1, 2, 3, 4)
        >>> datapipe = create_chained_datapipe(
        ...     [
        ...         {
        ...             "_target_": "torch.utils.data.datapipes.iter.IterableWrapper",
        ...             "iterable": [1, 2, 3, 4],
        ...         },
        ...         {"_target_": "torch.utils.data.datapipes.iter.Batcher", "batch_size": 2},
        ...     ]
        ... )
        >>> tuple(datapipe)
        ([1, 2], [3, 4])
    """
    if not configs:
        raise RuntimeError("It is not possible to create a DataPipe because the configs are empty")
    datapipe = factory(**configs[0])
    for config in configs[1:]:
        config = config.copy()  # Make a copy because the dict is modified below.
        target = config.pop(OBJECT_TARGET)
        datapipe = factory(target, datapipe, **config)
    return datapipe


def is_datapipe_config(config: dict) -> bool:
    r"""Indicate if the input configuration is a configuration for a
    ``DataPipe``.

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
            for a ``DataPipe`` object.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.datapipes import is_datapipe_config
        >>> is_datapipe_config(
        ...     {
        ...         "_target_": "torch.utils.data.datapipes.iter.IterableWrapper",
        ...         "iterable": [1, 2, 3, 4],
        ...     }
        ... )
        True
        >>> is_datapipe_config(
        ...     {
        ...         "_target_": "torch.utils.data.datapipes.map.SequenceWrapper",
        ...         "sequence": [1, 2, 3, 4],
        ...     }
        ... )
        True
    """
    return is_iter_datapipe_config(config) or is_map_datapipe_config(config)


def setup_datapipe(datapipe: IterDataPipe | MapDataPipe | dict) -> IterDataPipe | MapDataPipe:
    r"""Sets up a ``torch.utils.data.graph.DataPipe`` object.

    Args:
    ----
        datapipe (``IterDataPipe`` or ``MapDataPipe`` or dict):
            Specifies the datapipe or its configuration (dictionary).

    Returns:
    -------
        ``IterDataPipe`` or ``MapDataPipe``: The instantiated
            ``DataPipe`` object.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.datapipes import setup_datapipe
        >>> datapipe = setup_datapipe(
        ...     {
        ...         "_target_": "torch.utils.data.datapipes.iter.IterableWrapper",
        ...         "iterable": [1, 2, 3, 4],
        ...     }
        ... )
        >>> datapipe
        IterableWrapperIterDataPipe
        >>> setup_datapipe(datapipe)  # Do nothing because the datapipe is already instantiated
        IterableWrapperIterDataPipe
    """
    if isinstance(datapipe, dict):
        logger.info(
            "Initializing a `torch.utils.data.graph.DataPipe` from its configuration... "
            f"{str_target_object(datapipe)}"
        )
        datapipe = factory(**datapipe)
    return datapipe
