r"""This module defines the optimizer base class."""

__all__ = ["setup_optimizer"]

import logging
from typing import Optional, Union

from objectory import factory
from torch.nn import Module
from torch.optim import Optimizer

from gravitorch.nn.utils.module_helpers import has_learnable_parameters
from gravitorch.utils.format import str_target_object

logger = logging.getLogger(__name__)


def setup_optimizer(model: Module, optimizer: Union[Optimizer, dict, None]) -> Optional[Optimizer]:
    r"""Sets up the optimizer.

    The optimizer is instantiated from its configuration by using the
    ``OptimizerFactory`` factory function.

    Args:
        model (``torch.nn.Module``): Specifies the model to train.
        optimizer (``torch.optim.Optimizer`` or dict or ``None``):
            Specifies the optimizer or its configuration. If ``None``,
            no optimizer is instantiated.

    Returns:
        ``torch.optim.Optimizer`` or ``None``: The (instantiated)
            optimizer if the input is not ``None``, otherwise ``None``.

    Example usage:

    .. code-block:: python

        >>> from gravitorch.optimizers import setup_optimizer
        >>> import torch
        >>> optimizer = setup_optimizer(
        ...     torch.nn.Linear(4, 6),
        ...     optimizer={'_target_': "torch.optim.SGD", "lr": 0.01}
        ... )
        >>> optimizer
        SGD (
        Parameter Group 0
            dampening: 0
            foreach: None
            lr: 0.01
            maximize: False
            momentum: 0
            nesterov: False
            weight_decay: 0
        )
        >>> model = torch.nn.Linear(4, 6)
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        >>> setup_optimizer(model, optimizer)
        SGD (
        Parameter Group 0
            dampening: 0
            foreach: None
            lr: 0.001
            maximize: False
            momentum: 0
            nesterov: False
            weight_decay: 0
        )
    """
    if optimizer is None:
        logger.info("No optimizer")
        return None
    if not has_learnable_parameters(model):
        logger.info(
            "The optimizer is not initialized because the model "
            "does not have learnable parameters"
        )
        return None
    if isinstance(optimizer, dict):
        logger.info(
            f"Initializing an optimizer from its configuration... {str_target_object(optimizer)}"
        )
        optimizer = factory(params=model.parameters(), **optimizer)
    else:
        logger.info(
            "The optimizer is already created. The optimizer has to be initialized "
            "with the model parameters."
        )
    return optimizer
