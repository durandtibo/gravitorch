from __future__ import annotations

__all__ = ["setup_lr_scheduler_creator"]

import logging

from gravitorch.creators.lr_scheduler.base import BaseLRSchedulerCreator
from gravitorch.creators.lr_scheduler.vanilla import VanillaLRSchedulerCreator
from gravitorch.utils.format import str_target_object

logger = logging.getLogger(__name__)


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
        ...         "_target_": "gravitorch.creators.lr_scheduler.VanillaLRSchedulerCreator",
        ...         "lr_scheduler_config": {
        ...             "_target_": "torch.optim.lr_scheduler.StepLR",
        ...             "step_size": 5,
        ...         },
        ...     }
        ... )
        >>> creator
        VanillaLRSchedulerCreator(
          (lr_scheduler_config): {'_target_': 'torch.optim.lr_scheduler.StepLR', 'step_size': 5}
          (lr_scheduler_handler): None
          (add_module_to_engine): True
        )
    """
    if creator is None:
        creator = VanillaLRSchedulerCreator()
    if isinstance(creator, dict):
        logger.info(
            "Initializing the LR scheduler creator from its configuration... "
            f"{str_target_object(creator)}"
        )
        creator = BaseLRSchedulerCreator.factory(**creator)
    return creator
