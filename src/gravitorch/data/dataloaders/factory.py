from __future__ import annotations

__all__ = [
    "create_dataloader",
    "create_dataloader2",
    "is_dataloader_config",
    "is_dataloader2_config",
    "setup_dataloader",
    "setup_dataloader2",
]

import logging
from collections.abc import Iterable

from objectory import factory
from objectory.utils import is_object_config
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.graph import DataPipe

from gravitorch.data.datasets.factory import setup_dataset
from gravitorch.utils.factory import setup_object
from gravitorch.utils.format import str_target_object
from gravitorch.utils.imports import check_torchdata, is_torchdata_available

if is_torchdata_available():
    from torchdata.dataloader2 import DataLoader2, ReadingServiceInterface
    from torchdata.dataloader2.adapter import Adapter
else:  # pragma: no cover
    Adapter = "Adapter"
    DataLoader2 = "DataLoader2"
    ReadingServiceInterface = "ReadingServiceInterface"

logger = logging.getLogger(__name__)


def create_dataloader(dataset: Dataset | dict, **kwargs) -> DataLoader:
    r"""Instantiates a ``torch.utils.data.DataLoader`` from a
    ``torch.utils.data.Dataset`` or its configuration.

    Args:
        dataset (``torch.utils.data.Dataset`` or ``dict``): Specifies
            a dataset or its configuration.
        **kwargs: See ``torch.utils.data.DataLoader`` documentation.

    Returns:
        ``torch.utils.data.DataLoader``: The instantiated dataloader.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.data.dataloaders import create_dataloader
        >>> create_dataloader(
        ...     {
        ...         "_target_": "gravitorch.data.datasets.DummyMultiClassDataset",
        ...         "num_examples": 10,
        ...         "num_classes": 2,
        ...         "feature_size": 4,
        ...     }
        ... )
        <torch.utils.data.dataloader.DataLoader at 0x0123456789>
    """
    return DataLoader(
        setup_dataset(dataset), **{key: setup_object(value) for key, value in kwargs.items()}
    )


def create_dataloader2(
    datapipe: DataPipe | dict,
    datapipe_adapter_fn: Iterable[Adapter | dict] | Adapter | dict | None = None,
    reading_service: ReadingServiceInterface | dict | None = None,
) -> DataLoader2:
    r"""Instantiates a ``torchdata.dataloader2.DataLoader2`` object
    from a ``DataPipe`` or its configuration.

    Args:
        datapipe: Specifies the DataPipe or its configuration.
        datapipe_adapter_fn: Specifies the ``Adapter`` function(s) that
            will be applied to the DataPipe. Default: ``None``
        reading_service: Defines how ``DataLoader2`` should execute
            operations over the ``DataPipe``, e.g.
            multiprocessing/distributed. Default: ``None``

    Returns:
        ``torchdata.dataloader2.DataLoader2``: The instantiated
            ``DataLoader2``

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.data.dataloaders import create_dataloader2
        >>> create_dataloader2(
        ...     {
        ...         "_target_": "torch.utils.data.datapipes.iter.IterableWrapper",
        ...         "iterable": range(10),
        ...     }
        ... )
        <torchdata.dataloader2.dataloader2.DataLoader2 at 0x0123456789>
    """
    check_torchdata()
    return DataLoader2(
        datapipe=setup_object(datapipe),
        datapipe_adapter_fn=setup_object(datapipe_adapter_fn),
        reading_service=setup_object(reading_service),
    )


def is_dataloader_config(config: dict) -> bool:
    r"""Indicate if the input configuration is a configuration for a
    ``torch.utils.data.DataLoader``.

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
            for a ``torch.utils.data.DataLoader`` object.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.data.dataloaders import is_dataloader_config
        >>> is_dataloader_config({"_target_": "torch.utils.data.DataLoader"})
        True
    """
    return is_object_config(config, DataLoader)


def is_dataloader2_config(config: dict) -> bool:
    r"""Indicate if the input configuration is a configuration for a
    ``torchdata.dataloader2.DataLoader2``.

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
            for a ``torchdata.dataloader2.DataLoader2`` object.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.data.dataloaders import is_dataloader2_config
        >>> is_dataloader2_config({"_target_": "torchdata.dataloader2.DataLoader2"})
        True
    """
    check_torchdata()
    return is_object_config(config, DataLoader2)


def setup_dataloader(dataloader: DataLoader | dict) -> DataLoader:
    r"""Sets up a ``torch.utils.data.DataLoader`` object.

    Args:
    ----
        dataloader (``torch.utils.data.DataLoader`` or dict):
            Specifies the dataloader or its configuration (dictionary).

    Returns:
    -------
        ``torch.utils.data.DataLoader``: The instantiated dataloader.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.data.dataloaders import setup_dataloader
        >>> from gravitorch.data.datasets import ExampleDataset
        >>> dataloader = setup_dataloader(
        ...     {"_target_": "torch.utils.data.DataLoader", "dataset": ExampleDataset((1, 2, 3, 4))}
        ... )
        >>> dataloader
        <torch.utils.data.dataloader.DataLoader at 0x119bd42e0>
        >>> setup_dataloader(
        ...     dataloader
        ... )  # Do nothing because the dataloader is already instantiated
        <torch.utils.data.dataloader.DataLoader at 0x119bd42e0>
    """
    if isinstance(dataloader, dict):
        logger.info(
            "Initializing a `torch.utils.data.DataLoader` from its configuration... "
            f"{str_target_object(dataloader)}"
        )
        dataloader = factory(**dataloader)
    return dataloader


def setup_dataloader2(dataloader: DataLoader2 | dict) -> DataLoader2:
    r"""Sets up a ``torchdata.dataloader2.DataLoader2`` object.

    Args:
    ----
        dataloader (``torchdata.dataloader2.DataLoader2`` or dict):
            Specifies the dataloader or its configuration (dictionary).

    Returns:
    -------
        ``torchdata.dataloader2.DataLoader2``: The instantiated dataloader.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.data.dataloaders import setup_dataloader2
        >>> from torch.utils.data.datapipes.iter import IterableWrapper
        >>> dataloader = setup_dataloader2(
        ...     {
        ...         "_target_": "torchdata.dataloader2.DataLoader2",
        ...         "datapipe": IterableWrapper((1, 2, 3, 4)),
        ...     }
        ... )
        >>> dataloader
        <torch.utils.data.dataloader.DataLoader at 0x119bd42e0>
        >>> setup_dataloader(
        ...     dataloader
        ... )  # Do nothing because the dataloader is already instantiated
        <torch.utils.data.dataloader.DataLoader at 0x119bd42e0>
    """
    check_torchdata()
    if isinstance(dataloader, dict):
        logger.info(
            "Initializing a `torchdata.dataloader2.DataLoader2` from its configuration... "
            f"{str_target_object(dataloader)}"
        )
        dataloader = factory(**dataloader)
    return dataloader
