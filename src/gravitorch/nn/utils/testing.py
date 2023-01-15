__all__ = ["is_loss_decreasing", "is_loss_decreasing_with_adam", "is_loss_decreasing_with_sgd"]

import logging
from collections.abc import Callable
from typing import Union

from torch import Tensor
from torch.nn import Module
from torch.nn.utils import clip_grad_norm_
from torch.optim import SGD, Adam, Optimizer

from gravitorch.nn.utils.module_helpers import module_mode
from gravitorch.utils.seed import torch_seed

logger = logging.getLogger(__name__)


def is_loss_decreasing(
    module: Module,
    criterion: Union[Module, Callable[[Tensor, Tensor], Tensor]],
    optimizer: Optimizer,
    feature: Tensor,
    target: Tensor,
    num_iterations: int = 1,
    random_seed: int = 10772155803920552556,
) -> bool:
    r"""Checks if the loss decreased after some iterations.

    Args:
        module (``torch.nn.Module``): Specifies the module to test.
        criterion (``torch.nn.Module`` or callable): Specifies the
            criterion.
        optimizer (``torch.optim.Optimizer``): Specifies the optimizer
            to update the weights of the model.
        feature (``torch.Tensor``): Specifies the input of the module.
        target (``torch.Tensor``): Specifies the target used to
            compute the loss.
        num_iterations (int, optional): Specifies the number of
            optimization steps. Default: ``1``
        random_seed (int, optional): Specifies the random seed.
            The random seed is used to have a deterministic when the
            loss value is computed. Default: ``10772155803920552556``

    Returns:
        bool: ``True`` if the loss decreased after some iterations,
            otherwise ``False``.

    Example usage:

    .. code-block:: python

        >>> import torch
        >>> from torch import nn
        >>> from gravitorch.nn.utils import is_loss_decreasing
        >>> module = nn.Linear(4, 2)
        >>> is_loss_decreasing(
        ...     module=module,
        ...     criterion=nn.MSELoss(),
        ...     optimizer=SGD(module.parameters(), lr=0.01),
        ...     feature=torch.rand(4, 4),
        ...     target=torch.rand(4, 2),
        ... )
        True
    """
    with module_mode(module):
        module.eval()
        with torch_seed(random_seed):
            initial_loss = criterion(module(feature), target).item()
        logger.debug(f"Initial loss value: {initial_loss}")

        module.train()
        for _ in range(num_iterations):
            optimizer.zero_grad()
            loss = criterion(module(feature), target)
            loss.backward()
            clip_grad_norm_(module.parameters(), max_norm=1, norm_type=2)
            optimizer.step()

        module.eval()
        with torch_seed(random_seed):
            final_loss = criterion(module(feature), target).item()
        logger.debug(f"Final loss value: {final_loss}")
    return final_loss < initial_loss


def is_loss_decreasing_with_adam(
    module: Module,
    criterion: Union[Module, Callable[[Tensor, Tensor], Tensor]],
    feature: Tensor,
    target: Tensor,
    lr: float = 0.0003,
    num_iterations: int = 1,
    random_seed: int = 10772155803920552556,
) -> bool:
    r"""Checks if the loss decreased after some iterations.

    The module is trained with the Adam optimizer.

    Args:
        module (``torch.nn.Module``): Specifies the module to test.
        criterion (``torch.nn.Module`` or callable): Specifies the
            criterion.
        feature (``torch.Tensor``): Specifies the input of the module.
        target (``torch.Tensor``): Specifies the target used to
            compute the loss.
        lr (float, optional): Specifies the learning rate.
            Default: ``0.0003``
        num_iterations (int, optional): Specifies the number of
            optimization steps. Default: ``1``
        random_seed (int, optional): Specifies the random seed.
            The random seed is used to have a deterministic when
            the loss value is computed.
            Default: ``10772155803920552556``

    Returns:
        bool: ``True`` if the loss decreased after some iterations,
            otherwise ``False``.

    Example usage:

    .. code-block:: python

        >>> import torch
        >>> from torch import nn
        >>> from gravitorch.nn.utils import is_loss_decreasing_with_adam
        >>> is_loss_decreasing_with_adam(
        ...     module=nn.Linear(4, 2),
        ...     criterion=nn.MSELoss(),
        ...     feature=torch.rand(4, 4),
        ...     target=torch.rand(4, 2),
        ...     lr=0.0003,
        ... )
        True
    """
    return is_loss_decreasing(
        module=module,
        criterion=criterion,
        optimizer=Adam(module.parameters(), lr=lr),
        feature=feature,
        target=target,
        num_iterations=num_iterations,
        random_seed=random_seed,
    )


def is_loss_decreasing_with_sgd(
    module: Module,
    criterion: Union[Module, Callable[[Tensor, Tensor], Tensor]],
    feature: Tensor,
    target: Tensor,
    lr: float = 0.01,
    num_iterations: int = 1,
    random_seed: int = 10772155803920552556,
) -> bool:
    r"""Checks if the loss decreased after some iterations.

    The module is trained with the ``torch.optim.SGD`` optimizer.

    Args:
        module (``torch.nn.Module``): Specifies the module to test.
        criterion (``torch.nn.Module`` or callable): Specifies the
            criterion.
        feature (``torch.Tensor``): Specifies the input of the module.
        target (``torch.Tensor``): Specifies the target used to
            compute the loss.
        lr (float, optional): Specifies the learning rate.
            Default: ``0.01``
        num_iterations (int, optional): Specifies the number of
            optimization steps. Default: ``1``
        random_seed (int, optional): Specifies the random seed. The
        random seed is used to have a deterministic when the loss
            value is computed. Default: ``10772155803920552556``

    Returns:
        bool: ``True`` if the loss decreased after some iterations,
            otherwise ``False``.

    Example usage:

    .. code-block:: python

        >>> import torch
        >>> from torch import nn
        >>> from gravitorch.nn.utils import is_loss_decreasing_with_adam
        >>> is_loss_decreasing_with_adam(
        ...     module=nn.Linear(4, 2),
        ...     criterion=nn.MSELoss(),
        ...     feature=torch.rand(4, 4),
        ...     target=torch.rand(4, 2),
        ...     lr=0.01,
        ... )
        True
    """
    return is_loss_decreasing(
        module=module,
        criterion=criterion,
        optimizer=SGD(module.parameters(), lr=lr),
        feature=feature,
        target=target,
        num_iterations=num_iterations,
        random_seed=random_seed,
    )
