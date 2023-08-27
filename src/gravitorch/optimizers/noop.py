from __future__ import annotations

__all__ = ["NoOpOptimizer"]

from collections.abc import Callable, Iterable

from torch import Tensor
from torch.optim import Optimizer


class NoOpOptimizer(Optimizer):
    r"""Implements a no-op optimizer.

    This optimizer cannot be used to train a model. This optimizer can
    be used to simulate an optimizer that does not update the model
    parameters.

    Args:
    ----
        params: This input is not used. It is here to make it
            compatible with the other optimizers.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.optimizers import NoOpOptimizer
        >>> import torch
        >>> optimizer = NoOpOptimizer(torch.nn.Linear(4, 6).parameters())
        >>> optimizer
        NoOpOptimizer()
    """

    def __init__(self, params: Iterable[Tensor] | Iterable[dict]) -> None:
        r"""Do nothing."""

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def load_state_dict(self, state_dict: dict) -> None:
        r"""Do nothing."""

    def state_dict(self) -> dict:
        return {}

    def step(self, closure: Callable | None = None) -> None:
        r"""Do nothing."""
