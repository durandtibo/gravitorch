r"""This module defines some utility functions for the data loader
creators."""

from __future__ import annotations

__all__ = ["setup_dataloader_creator"]

import logging

from gravitorch.creators.dataloader.base import BaseDataLoaderCreator
from gravitorch.creators.dataloader.pytorch import AutoDataLoaderCreator
from gravitorch.utils.format import str_target_object

logger = logging.getLogger(__name__)


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
    return creator
