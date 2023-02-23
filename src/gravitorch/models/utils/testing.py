__all__ = ["is_loss_decreasing", "is_loss_decreasing_with_adam", "is_loss_decreasing_with_sgd"]

import logging
from typing import Any

from torch.nn.utils import clip_grad_norm_
from torch.optim import SGD, Adam, Optimizer

from gravitorch import constants as ct
from gravitorch.models.base import BaseModel
from gravitorch.nn.utils.module_helpers import module_mode
from gravitorch.utils.seed import torch_seed

logger = logging.getLogger(__name__)


def is_loss_decreasing(
    model: BaseModel,
    optimizer: Optimizer,
    batch: Any,
    num_iterations: int = 1,
    random_seed: int = 11409583194270223596,
) -> bool:
    r"""Checks if the loss decreased after some iterations.

    Args:
        model (``BaseModel``): Specifies the model to test.
        optimizer (``torch.optim.Optimizer``): Specifies the optimizer
            to update the weights of the model.
        batch: Specifies the batch used to train the model.
        num_iterations (int, optional): Specifies the number of
            optimization steps. Default: ``1``
        random_seed (int, optional): Specifies the random seed. The
            random seed is used to have a deterministic when the loss
            value is computed. Default: ``11409583194270223596``

    Returns:
        bool: ``True`` if the loss decreased after some iterations,
            otherwise ``False``.

    Example usage:

    .. code-block:: python

        >>> import torch
        >>> from torch import nn
        >>> from torch.optim import SGD
        >>> from gravitorch.models import VanillaModel
        >>> from gravitorch.models.criteria import VanillaLoss
        >>> from gravitorch.models.networks import BetaMLP
        >>> from gravitorch.models.utils import is_loss_decreasing
        >>> model = VanillaModel(
        ...     network=BetaMLP(input_size=6, hidden_sizes=(8, 4)),
        ...     criterion=VanillaLoss(criterion=nn.MSELoss()),
        ... )
        >>> optimizer = SGD(model.parameters(), lr=0.01)
        >>> is_loss_decreasing(
        ...     model=model,
        ...     optimizer=optimizer,
        ...     batch={'input': torch.randn(2, 6), 'target': torch.randn(2, 4)},
        ... )
        True
    """
    with module_mode(model):
        model.eval()
        with torch_seed(random_seed):
            initial_loss = model(batch)[ct.LOSS].item()
        logger.debug(f"Initial loss value: {initial_loss}")

        model.train()
        for _ in range(num_iterations):
            optimizer.zero_grad()
            output = model(batch)
            output[ct.LOSS].backward()
            clip_grad_norm_(model.parameters(), max_norm=1, norm_type=2)
            optimizer.step()

        model.eval()
        with torch_seed(random_seed):
            final_loss = model(batch)[ct.LOSS].item()
        logger.debug(f"Final loss value: {final_loss}")
    return final_loss < initial_loss


def is_loss_decreasing_with_adam(
    model: BaseModel,
    batch: Any,
    lr: float = 0.0003,
    num_iterations: int = 1,
    random_seed: int = 11409583194270223596,
) -> bool:
    r"""Checks if the loss decreased after some iterations.

    The weights of the model are updated with Adam.

    Args:
        model (``BaseModel``): Specifies the model to test.
        batch: Specifies the batch used to train the model.
        lr (float, optional): Specifies the learning rate.
            Default: ``0.0003``
        num_iterations (int, optional): Specifies the number of
            optimization steps. Default: ``1``
        random_seed (int, optional): Specifies the random seed. The
            random seed is used to have a deterministic when the loss
            value is computed. Default: ``11409583194270223596``

    Returns:
        bool: ``True`` if the loss decreased after some iterations,
            otherwise ``False``.

    Example usage:

    .. code-block:: python

        >>> import torch
        >>> from torch import nn
        >>> from gravitorch.models import VanillaModel
        >>> from gravitorch.models.criteria import VanillaLoss
        >>> from gravitorch.models.networks import BetaMLP
        >>> from gravitorch.models.utils import is_loss_decreasing_with_adam
        >>> is_loss_decreasing_with_adam(
        ...     model=VanillaModel(
        ...         network=BetaMLP(input_size=6, hidden_sizes=(8, 4)),
        ...         criterion=VanillaLoss(criterion=nn.MSELoss()),
        ...     ),
        ...     lr=0.0003,
        ...     batch={'input': torch.randn(2, 6), 'target': torch.randn(2, 4)},
        ... )
        True
    """
    return is_loss_decreasing(
        model=model,
        optimizer=Adam(model.parameters(), lr=lr),
        batch=batch,
        num_iterations=num_iterations,
        random_seed=random_seed,
    )


def is_loss_decreasing_with_sgd(
    model: BaseModel,
    batch: Any,
    lr: float = 0.01,
    num_iterations: int = 1,
    random_seed: int = 11409583194270223596,
) -> bool:
    r"""Checks if the loss decreased after some iterations.

    The weights of the model are updated with SGD.

    Args:
        model (``BaseModel``): Specifies the model to test.
        batch: Specifies the batch used to train the model.
        lr (float, optional): Specifies the learning rate. Default: ``0.01``
        num_iterations (int, optional): Specifies the number of
            optimization steps. Default: ``1``
        random_seed (int, optional): Specifies the random seed. The
            random seed is used to have a deterministic when the loss
            value is computed. Default: ``11409583194270223596``

    Returns:
        bool: ``True`` if the loss decreased after some iterations,
            otherwise ``False``.

    Example usage:

    .. code-block:: python

        >>> import torch
        >>> from torch import nn
        >>> from gravitorch.models import VanillaModel
        >>> from gravitorch.models.criteria import VanillaLoss
        >>> from gravitorch.models.networks import BetaMLP
        >>> from gravitorch.models.utils import is_loss_decreasing_with_sgd
        >>> is_loss_decreasing_with_sgd(
        ...     model=VanillaModel(
        ...         network=BetaMLP(input_size=6, hidden_sizes=(8, 4)),
        ...         criterion=VanillaLoss(criterion=nn.MSELoss()),
        ...     ),
        ...     lr=0.01,
        ...     batch={'input': torch.randn(2, 6), 'target': torch.randn(2, 4)},
        ... )
        True
    """
    return is_loss_decreasing(
        model=model,
        optimizer=SGD(model.parameters(), lr=lr),
        batch=batch,
        num_iterations=num_iterations,
        random_seed=random_seed,
    )
