from __future__ import annotations

__all__ = ["create_dataloader", "is_dataloader_config", "setup_dataloader"]

import logging

from objectory import factory
from objectory.utils import is_object_config
from torch.utils.data import DataLoader, Dataset

from gravitorch.datasets import setup_dataset
from gravitorch.utils.factory import setup_object
from gravitorch.utils.format import str_target_object

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

        >>> from gravitorch.dataloaders import create_dataloader
        >>> create_dataloader(
        ...     {
        ...         "_target_": "gravitorch.datasets.DummyMultiClassDataset",
        ...         "num_examples": 10,
        ...         "num_classes": 2,
        ...         "feature_size": 4,
        ...     }
        ... )
        <torch.utils.data.dataloader.DataLoader object at 0x...>
    """
    return DataLoader(
        setup_dataset(dataset), **{key: setup_object(value) for key, value in kwargs.items()}
    )


def is_dataloader_config(config: dict) -> bool:
    r"""Indicates if the input configuration is a configuration for a
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

        >>> from gravitorch.dataloaders import is_dataloader_config
        >>> is_dataloader_config({"_target_": "torch.utils.data.DataLoader"})
        True
    """
    return is_object_config(config, DataLoader)


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

        >>> from gravitorch.dataloaders import setup_dataloader
        >>> from gravitorch.datasets import ExampleDataset
        >>> dataloader = setup_dataloader(
        ...     {"_target_": "torch.utils.data.DataLoader", "dataset": ExampleDataset((1, 2, 3, 4))}
        ... )
        >>> dataloader
        <torch.utils.data.dataloader.DataLoader object at 0x...>
    """
    if isinstance(dataloader, dict):
        logger.info(
            "Initializing a `torch.utils.data.DataLoader` from its configuration... "
            f"{str_target_object(dataloader)}"
        )
        dataloader = factory(**dataloader)
    return dataloader
