from __future__ import annotations

__all__ = ["BaseInitializer"]

from abc import ABC, abstractmethod

from objectory import AbstractFactory
from torch.nn import Module


class BaseInitializer(ABC, metaclass=AbstractFactory):
    r"""Defines the base parameter initializer.

    Note that there are other ways to initialize the model parameters.
    For example, you can initialize the weights of your model directly
    in the model.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.nn.init import Constant
        >>> from torch import nn
        >>> module = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.BatchNorm1d(6), nn.Linear(6, 1))
        >>> initializer = Constant(value=2)
        >>> initializer
        Constant(value=2.0, learnable_only=True, log_info=False)
        >>> initializer.initialize(module)
        >>> for key, param in module.named_parameters():
        ...     print(key, param)
        ...
        0.weight Parameter containing:
        tensor([[2., 2., 2., 2.],
                [2., 2., 2., 2.],
                [2., 2., 2., 2.],
                [2., 2., 2., 2.],
                [2., 2., 2., 2.],
                [2., 2., 2., 2.]], requires_grad=True)
        0.bias Parameter containing:
        tensor([2., 2., 2., 2., 2., 2.], requires_grad=True)
        2.weight Parameter containing:
        tensor([2., 2., 2., 2., 2., 2.], requires_grad=True)
        2.bias Parameter containing:
        tensor([2., 2., 2., 2., 2., 2.], requires_grad=True)
        3.weight Parameter containing:
        tensor([[2., 2., 2., 2., 2., 2.]], requires_grad=True)
        3.bias Parameter containing:
        tensor([2.], requires_grad=True)
    """

    @abstractmethod
    def initialize(self, module: Module) -> None:
        r"""Initializes the parameters of the model.

            The parameters of the model should be updated in-place.

            Args:
            ----
                module (``torch.nn.Module``): Specifies the module to
                    initialize.

            Example usage:

        .. code-block:: pycon

            >>> from gravitorch.nn.init import Constant
            >>> from torch import nn
            >>> module = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.BatchNorm1d(6), nn.Linear(6, 1))
            >>> initializer = Constant(value=2)
            >>> initializer.initialize(module)
            >>> for key, param in module.named_parameters():
            ...     print(key, param)
            ...
            0.weight Parameter containing:
            tensor([[2., 2., 2., 2.],
                    [2., 2., 2., 2.],
                    [2., 2., 2., 2.],
                    [2., 2., 2., 2.],
                    [2., 2., 2., 2.],
                    [2., 2., 2., 2.]], requires_grad=True)
            0.bias Parameter containing:
            tensor([2., 2., 2., 2., 2., 2.], requires_grad=True)
            2.weight Parameter containing:
            tensor([2., 2., 2., 2., 2., 2.], requires_grad=True)
            2.bias Parameter containing:
            tensor([2., 2., 2., 2., 2., 2.], requires_grad=True)
            3.weight Parameter containing:
            tensor([[2., 2., 2., 2., 2., 2.]], requires_grad=True)
            3.bias Parameter containing:
            tensor([2.], requires_grad=True)
        """
