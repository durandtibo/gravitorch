from __future__ import annotations

__all__ = ["setup_optimizer_creator"]

import logging

from gravitorch.creators.optimizer.base import BaseOptimizerCreator
from gravitorch.creators.optimizer.noo import NoOptimizerCreator
from gravitorch.utils.format import str_target_object

logger = logging.getLogger(__name__)


def setup_optimizer_creator(creator: BaseOptimizerCreator | dict | None) -> BaseOptimizerCreator:
    r"""Sets up the optimizer creator.

    The optimizer creator is instantiated from its configuration
    by using the ``BaseOptimizerCreator`` factory function.

    Args:
    ----
        creator (``BaseOptimizerCreator`` or dict or ``None``):
            Specifies the optimizer creator or its configuration.
            If ``None``, a ``NoOptimizerCreator`` is created.

    Returns:
    -------
        ``BaseOptimizerCreator``: The instantiated optimizer creator.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.testing import create_dummy_engine, DummyClassificationModel
        >>> from gravitorch.creators.optimizer import setup_optimizer_creator
        >>> creator = setup_optimizer_creator(
        ...     {
        ...         "_target_": "gravitorch.creators.optimizer.VanillaOptimizerCreator",
        ...         "optimizer_config": {"_target_": "torch.optim.SGD", "lr": 0.01},
        ...     }
        ... )
        >>> creator
        VanillaOptimizerCreator(add_module_to_engine=True)
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
