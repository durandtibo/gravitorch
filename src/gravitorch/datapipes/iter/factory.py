from __future__ import annotations

__all__ = ["create_sequential_iterdatapipe", "is_iterdatapipe_config", "setup_iterdatapipe"]

from collections.abc import Sequence

from objectory import OBJECT_TARGET, factory
from objectory.utils import is_object_config
from torch.utils.data import IterDataPipe


def create_sequential_iterdatapipe(configs: Sequence[dict]) -> IterDataPipe:
    r"""Creates a sequence of ``IterDataPipe``s from their configuration.

    Args:
    ----
        configs: Specifies the config of each ``IterDataPipe`` in the
            sequence. The configs sequence follows the order of the
            ``IterDataPipe``s. The first config is used to create the
            first ``IterDataPipe`` (a.k.a. source), and the last
            config is used to create the last ``IterDataPipe``
            (a.k.a. sink). This function assumes all the DataPipes
            have a single source DataPipe as their first argument,
            excepts for the first DataPipe.

    Returns:
    -------
        ``IterDataPipe``: The last ``IterDataPipe`` of the sequence
            (a.k.a. sink).

    Raises:
    ------
        ValueError if the ``IterDataPipe`` configuration sequence is
            empty.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.datapipes.iter.factory import create_sequential_iterdatapipe
        >>> datapipe = create_sequential_iterdatapipe(
        ...     [
        ...         {
        ...             "_target_": "gravitorch.datapipes.iter.SourceWrapper",
        ...             "data": [1, 2, 3, 4],
        ...         },
        ...     ],
        ... )
        >>> tuple(datapipe)
        (1, 2, 3, 4)
        >>> datapipe = create_sequential_iterdatapipe(
        ...     [
        ...         {
        ...             "_target_": "gravitorch.datapipes.iter.SourceWrapper",
        ...             "data": [1, 2, 3, 4],
        ...         },
        ...         {"_target_": "torch.utils.data.datapipes.iter.Batcher", "batch_size": 2},
        ...     ]
        ... )
        >>> tuple(datapipe)
        ([1, 2], [3, 4])
    """
    if not configs:
        raise ValueError("It is not possible to create a DataPipe because the configs are empty")
    datapipe = factory(**configs[0])
    for config in configs[1:]:
        config = config.copy()  # Make a copy because the dict is modified below.
        target = config.pop(OBJECT_TARGET)
        datapipe = factory(target, datapipe, **config)
    return datapipe


def is_iterdatapipe_config(config: dict) -> bool:
    r"""Indicate if the input configuration is a configuration for a
    ``IterDataPipe``.

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
            for a ``IterDataPipe`` object.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.datapipes import is_iterdatapipe_config
        >>> is_iterdatapipe_config(
        ...     {
        ...         "_target_": "torch.utils.data.datapipes.iter.IterableWrapper",
        ...         "iterable": [1, 2, 3, 4],
        ...     }
        ... )
        True
    """
    return is_object_config(config, IterDataPipe)


def setup_iterdatapipe(datapipe: IterDataPipe | Sequence[dict]) -> IterDataPipe:
    r"""Sets up an ``IterDataPipe``.

    Note: it is only possible to create a sequential DataPipe from
    its configuration. See ``create_sequential_iterdatapipe``
    documentation for more information.

    Args:
    ----
        datapipe: Specifies the DataPipe or its configuration.

    Returns:
    -------
        ``IterDataPipe``: The last ``IterDataPipe`` of the sequence
            (a.k.a. sink).

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.datapipes.iter import setup_iterdatapipe
        >>> datapipe = setup_iterdatapipe(
        ...     [
        ...         {
        ...             "_target_": "gravitorch.datapipes.iter.SourceWrapper",
        ...             "data": [1, 2, 3, 4],
        ...         },
        ...     ],
        ... )
        >>> tuple(datapipe)
        (1, 2, 3, 4)
        >>> datapipe = setup_iterdatapipe(
        ...     [
        ...         {
        ...             "_target_": "gravitorch.datapipes.iter.SourceWrapper",
        ...             "data": [1, 2, 3, 4],
        ...         },
        ...         {"_target_": "torch.utils.data.datapipes.iter.Batcher", "batch_size": 2},
        ...     ]
        ... )
        >>> tuple(datapipe)
        ([1, 2], [3, 4])
    """
    if isinstance(datapipe, IterDataPipe):
        return datapipe
    return create_sequential_iterdatapipe(datapipe)
