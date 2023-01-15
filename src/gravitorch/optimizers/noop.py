__all__ = ["NoOpOptimizer"]

from collections.abc import Iterable
from typing import Union

from torch import Tensor
from torch.optim import Optimizer


class NoOpOptimizer(Optimizer):
    r"""Implements a no-op optimizer.

    This optimizer cannot be used to train a model. This optimizer can
    be used to simulate an optimizer that does not update the model
    parameters.

    Args:
        params: This input is not used. It is here to make it
            compatible with the other optimizers.
    """

    def __init__(self, params: Union[Iterable[Tensor], Iterable[dict]]):
        r"""Do nothing."""

    def load_state_dict(self, state_dict: dict) -> None:
        r"""Do nothing."""

    def state_dict(self) -> dict:
        return {}

    def step(self, closure=None):
        r"""Do nothing."""
