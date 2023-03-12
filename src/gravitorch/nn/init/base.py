__all__ = ["BaseInitializer"]

from abc import ABC, abstractmethod

from objectory import AbstractFactory
from torch.nn import Module


class BaseInitializer(ABC, metaclass=AbstractFactory):
    r"""Defines the base parameter initializer.

    Note that there are other ways to initialize the model parameters.
    For example, you can initialize the weights of your model directly
    in the model.
    """

    @abstractmethod
    def initialize(self, module: Module) -> None:
        r"""Initializes the parameters of the model.

        The parameters of the model should be updated in-place.

        Args:
        ----
            module (``torch.nn.Module``): Specifies the module to
                initialize.
        """
