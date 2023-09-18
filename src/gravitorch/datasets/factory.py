r"""This module defines some functionalities to instantiate dynamically
a ``torch.utils.data.Dataset`` object from its configuration."""

from __future__ import annotations

__all__ = ["create_datasets", "is_dataset_config", "setup_dataset"]

import logging
from collections.abc import Hashable, Mapping
from typing import TypeVar

from objectory import factory
from objectory.utils import is_object_config
from torch.utils.data import Dataset

from gravitorch.utils.format import str_target_object

logger = logging.getLogger(__name__)

T = TypeVar("T")


def create_datasets(datasets: Mapping[Hashable, Dataset | dict]) -> dict[Hashable, Dataset]:
    r"""Create datasets indexed by dataset split.

    Args:
    ----
        datasets (``Mapping``): Specifies the datasets or their
            configuration. The key is the dataset split name and the
            value is the dataset or its configuration.

    Returns:
    -------
        dict: The instantiated datasets.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.datasets import ExampleDataset, create_datasets
        >>> create_datasets(
        ...     {
        ...         "train": ExampleDataset((1, 2, 3)),
        ...         "val": {
        ...             "_target_": "gravitorch.datasets.ExampleDataset",
        ...             "examples": (4, 5),
        ...         },
        ...     }
        ... )
        {'train': ExampleDataset(num_examples=3), 'val': ExampleDataset(num_examples=2)}
    """
    return {split: setup_dataset(dataset) for split, dataset in datasets.items()}


def is_dataset_config(config: dict) -> bool:
    r"""Indicates if the input configuration is a configuration for a
    ``torch.utils.data.Dataset``.

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
            for a ``torch.utils.data.Dataset`` object.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.datasets import is_dataset_config
        >>> is_dataset_config(
        ...     {"_target_": "gravitorch.datasets.ExampleDataset", "examples": [1, 2, 1, 3]}
        ... )
        True
    """
    return is_object_config(config, Dataset)


def setup_dataset(dataset: Dataset | dict | None) -> Dataset | None:
    r"""Sets up a dataset.

    Args:
    ----
        dataset (``Dataset`` or dict or ``None``): Specifies the
            dataset or its configuration (dictionary). If a
            configuration is given, a dataset object is instantiated
            from the configuration.

    Returns:
    -------
        ``torch.utils.data.Dataset`` or ``None``: A dataset object or
            ``None`` if there is no dataset.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.datasets import setup_dataset
        >>> dataset = setup_dataset(
        ...     {"_target_": "gravitorch.datasets.ExampleDataset", "examples": [1, 2, 1, 3]},
        ... )
        >>> dataset
        ExampleDataset(num_examples=4)
    """
    if isinstance(dataset, dict):
        logger.info(
            f"Initializing a dataset from its configuration... {str_target_object(dataset)}"
        )
        dataset = factory(**dataset)
    return dataset
