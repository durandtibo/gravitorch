r"""This module defines some utility functions for the data loader
creators."""

from __future__ import annotations

__all__ = ["is_dataloader_creator_config", "setup_dataloader_creator"]

import logging

from objectory.utils import is_object_config

from gravitorch.creators.dataloader.base import BaseDataLoaderCreator
from gravitorch.creators.dataloader.pytorch import AutoDataLoaderCreator
from gravitorch.utils.format import str_target_object

logger = logging.getLogger(__name__)


def is_dataloader_creator_config(config: dict) -> bool:
    r"""Indicates if the input configuration is a configuration for a
    ``BaseDataLoaderCreator``.

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
            for a ``BaseDataLoaderCreator`` object.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.creators.dataloader import is_dataloader_creator_config
        >>> is_dataloader_creator_config(
        ...     {"_target_": "gravitorch.creators.dataloader.DataLoaderCreator"}
        ... )
        True
    """
    return is_object_config(config, BaseDataLoaderCreator)


def setup_dataloader_creator(
    creator: BaseDataLoaderCreator | dict | None,
) -> BaseDataLoaderCreator:
    r"""Sets up a data loader creator.

    Args:
    ----
        creator (``BaseDataLoaderCreator`` or dict or None):
            Specifies the data loader creator or its configuration.
            If ``None``, a data loader creator will be created
            automatically.

    Returns:
    -------
        ``BaseDataLoaderCreator``: The data loader creator.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.creators.dataloader import setup_dataloader_creator
        >>> from gravitorch.testing import DummyDataset
        >>> creator = setup_dataloader_creator(
        ...     {"_target_": "gravitorch.creators.dataloader.DataLoaderCreator"}
        ... )
        >>> creator
        DataLoaderCreator(
          (seed): 0
        )
    """
    if creator is None:
        creator = AutoDataLoaderCreator()
    if isinstance(creator, dict):
        logger.info(
            "Initializing a data loader creator from its configuration... "
            f"{str_target_object(creator)}"
        )
        creator = BaseDataLoaderCreator.factory(**creator)
    if not isinstance(creator, BaseDataLoaderCreator):
        logger.warning(f"creator is not a `BaseDataLoaderCreator` (received: {type(creator)})")
    return creator
