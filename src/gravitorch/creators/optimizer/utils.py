__all__ = ["setup_optimizer_creator"]

import logging
from typing import Union

from gravitorch.creators.optimizer.base import BaseOptimizerCreator
from gravitorch.creators.optimizer.noo import NoOptimizerCreator
from gravitorch.utils.format import str_target_object

logger = logging.getLogger(__name__)


def setup_optimizer_creator(
    creator: Union[BaseOptimizerCreator, dict, None]
) -> BaseOptimizerCreator:
    r"""Sets up the optimizer creator.

    The optimizer creator is instantiated from its configuration
    by using the ``BaseOptimizerCreator`` factory function.

    Args:
        creator (``BaseOptimizerCreator`` or dict or ``None``):
            Specifies the optimizer creator or its configuration.
            If ``None``, a ``NoOptimizerCreator`` is created.

    Returns:
        ``BaseOptimizerCreator``: The instantiated optimizer creator.
    """
    if creator is None:
        creator = NoOptimizerCreator()
    if isinstance(creator, dict):
        logger.info(
            "Initializing the optimizer creator from its configuration... "
            f"{str_target_object(creator)}"
        )
        creator = BaseOptimizerCreator.factory(**creator)
    return creator
