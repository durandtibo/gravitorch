r"""This module defines some steps that are used in the experiment tracker."""

__all__ = ["Step", "EpochStep", "IterationStep"]

from dataclasses import dataclass


@dataclass
class Step:
    r"""Defines a generic step.

    A step should have a step number and a name.

    Args:
        step (int): Specifies the step number.
        name (str): Specifies the name.
    """
    step: int
    name: str


class EpochStep(Step):
    r"""Defines the epoch step.

    Args:
        step (int): Specifies the epoch number.
    """

    def __init__(self, epoch: int):
        super().__init__(step=epoch, name="epoch")


class IterationStep(Step):
    r"""Defines the iteration step.

    Args:
        step (int): Specifies the iteration number.
    """

    def __init__(self, iteration: int):
        super().__init__(step=iteration, name="iteration")
