from __future__ import annotations

__all__ = ["is_lr_scheduler_creator_config", "setup_lr_scheduler_creator"]

import logging

from objectory.utils import is_object_config

from gravitorch.creators.lr_scheduler.base import BaseLRSchedulerCreator
from gravitorch.creators.lr_scheduler.vanilla import LRSchedulerCreator
from gravitorch.utils.format import str_target_object

logger = logging.getLogger(__name__)


def is_lr_scheduler_creator_config(config: dict) -> bool:
    r"""Indicates if the input configuration is a configuration for a
    ``BaseLRSchedulerCreator``.

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
            for a ``BaseLRSchedulerCreator`` object.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.creators.lr_scheduler import is_lr_scheduler_creator_config
        >>> is_lr_scheduler_creator_config(
        ...     {
        ...         "_target_": "gravitorch.creators.lr_scheduler.LRSchedulerCreator",
        ...         "lr_scheduler_config": {
        ...             "_target_": "torch.optim.lr_scheduler.StepLR",
        ...             "step_size": 5,
        ...         },
        ...     }
        ... )
        True
    """
    return is_object_config(config, BaseLRSchedulerCreator)


def setup_lr_scheduler_creator(
    creator: BaseLRSchedulerCreator | dict | None,
) -> BaseLRSchedulerCreator:
    r"""Sets up the LR scheduler creator.

    The LR scheduler creator is instantiated from its configuration by
    using the ``BaseLRSchedulerCreator`` factory function.

    Args:
    ----
        creator (``BaseLRSchedulerCreator`` or dict or ``None``):
            Specifies the LR scheduler creator or its configuration.

    Returns:
    -------
        ``BaseLRSchedulerCreator``: The instantiated LR scheduler
            creator.

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from gravitorch.creators.lr_scheduler import setup_lr_scheduler_creator
        >>> creator = setup_lr_scheduler_creator(
        ...     {
        ...         "_target_": "gravitorch.creators.lr_scheduler.LRSchedulerCreator",
        ...         "lr_scheduler_config": {
        ...             "_target_": "torch.optim.lr_scheduler.StepLR",
        ...             "step_size": 5,
        ...         },
        ...     }
        ... )
        >>> creator
        LRSchedulerCreator(
          (lr_scheduler_config): {'_target_': 'torch.optim.lr_scheduler.StepLR', 'step_size': 5}
          (lr_scheduler_handler): None
          (add_module_to_engine): True
        )
    """
    if creator is None:
        creator = LRSchedulerCreator()
    if isinstance(creator, dict):
        logger.info(
            "Initializing the LR scheduler creator from its configuration... "
            f"{str_target_object(creator)}"
        )
        creator = BaseLRSchedulerCreator.factory(**creator)
    if not isinstance(creator, BaseLRSchedulerCreator):
        logger.warning(f"creator is not a `BaseLRSchedulerCreator` (received: {type(creator)})")
    return creator
