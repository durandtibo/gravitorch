r"""This module defines the base model."""

__all__ = ["BaseModel"]

from abc import abstractmethod
from typing import Any

from objectory import AbstractFactory
from torch.nn import Module


class BaseModel(Module, metaclass=AbstractFactory):
    r"""Defines the base model.

    To be compatible with the engine, the forward function of the model
    should return a dictionary. If you want to train the model, the
    output dictionary should contain the key ``'loss'`` with the loss
    value.
    """

    @abstractmethod
    def forward(self, batch: Any) -> dict:
        r"""Defines the forward function of the model that returns a dictionary
        containing the loss value.

        Args:
            batch: The input is the batch of data returned by the data
                loader.

        Returns:
            dict: dictionary containing the loss value.
        """
