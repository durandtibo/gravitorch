r"""This module defines some utility functions for the data loader creators."""

__all__ = ["setup_data_loader_creator"]

import logging
from typing import Union

from gravitorch.creators.dataloader.base import BaseDataLoaderCreator
from gravitorch.creators.dataloader.pytorch import AutoDataLoaderCreator
from gravitorch.utils.format import str_target_object

logger = logging.getLogger(__name__)


def setup_data_loader_creator(
    creator: Union[BaseDataLoaderCreator, dict, None]
) -> BaseDataLoaderCreator:
    r"""Sets up a data loader creator.

    Args:
        creator (``BaseDataLoaderCreator`` or dict or None):
            Specifies the data loader creator or its configuration.
            If ``None``, a data loader creator will be created
            automatically.

    Returns:
        ``BaseDataLoaderCreator``: The data loader creator.
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
