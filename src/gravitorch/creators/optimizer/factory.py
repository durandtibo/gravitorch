from __future__ import annotations

__all__ = ["is_optimizer_creator_config", "setup_optimizer_creator"]

import logging

from objectory.utils import is_object_config

from gravitorch.creators.optimizer.base import BaseOptimizerCreator
from gravitorch.creators.optimizer.noo import NoOptimizerCreator
from gravitorch.utils.format import str_target_object

logger = logging.getLogger(__name__)


def is_optimizer_creator_config(config: dict) -> bool:
    r"""Indicates if the input configuration is a configuration for a
    ``BaseOptimizerCreator``.

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
            for a ``BaseOptimizerCreator`` object.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.creators.optimizer import is_optimizer_creator_config
        >>> is_optimizer_creator_config(
        ...     {
        ...         "_target_": "gravitorch.creators.optimizer.OptimizerCreator",
        ...         "optimizer_config": {"_target_": "torch.optim.SGD", "lr": 0.01},
        ...     }
        ... )
        True
    """
    return is_object_config(config, BaseOptimizerCreator)


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
        ...         "_target_": "gravitorch.creators.optimizer.OptimizerCreator",
        ...         "optimizer_config": {"_target_": "torch.optim.SGD", "lr": 0.01},
        ...     }
        ... )
        >>> creator
        OptimizerCreator(add_module_to_engine=True)
    """
    if creator is None:
        creator = NoOptimizerCreator()
    if isinstance(creator, dict):
        logger.info(
            "Initializing the optimizer creator from its configuration... "
            f"{str_target_object(creator)}"
        )
        creator = BaseOptimizerCreator.factory(**creator)
    if not isinstance(creator, BaseOptimizerCreator):
        logger.warning(f"creator is not a `BaseOptimizerCreator` (received: {type(creator)})")
    return creator
