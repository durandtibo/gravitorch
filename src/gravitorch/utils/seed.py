"""Helper functions to help to reproduce experiments."""

__all__ = [
    "BaseRandomSeedSetter",
    "NumpyRandomSeedSetter",
    "RandomRandomSeedSetter",
    "RandomSeedSetter",
    "TorchRandomSeedSetter",
    "get_random_seed",
    "get_torch_generator",
    "manual_seed",
    "numpy_seed",
    "torch_seed",
]

import random
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Optional, Union

import numpy
import torch

from gravitorch.utils.format import str_indent, to_pretty_dict_str


def get_random_seed(seed: int) -> int:
    r"""Gets a random seed.

    Args:
        seed (int): Specifies a random seed to make the process
            reproducible.

    Returns:
        int: A random seed. The value is between ``-2 ** 63`` and
            ``2 ** 63 - 1``.

    Example usage:

    .. code-block:: python

        >>> from gravitorch.utils.seed import get_random_seed
        >>> get_random_seed(44)
        6176747449835261347
    """
    return torch.randint(
        -(2**63), 2**63 - 1, size=(1,), generator=get_torch_generator(seed)
    ).item()


def get_torch_generator(
    random_seed: int = 1, device: Union[torch.device, str, None] = "cpu"
) -> torch.Generator:
    r"""Creates a ``torch.Generator`` initialized with a given seed.

    Args:
        random_seed (int, optional): Specifies a random seed.
            Default: ``1``
        device (``torch.device`` or str or ``None``, optional):
            Specifies the desired device for the generator.
            Default: ``'cpu'``

    Returns:
        ``torch.Generator``

    Example usage:

    .. code-block:: python

        >>> import torch
        >>> from gravitorch.utils.seed import get_torch_generator
        >>> generator = get_torch_generator(42)
        >>> torch.rand(2, 4, generator=generator)
        tensor([[0.8823, 0.9150, 0.3829, 0.9593],
                [0.3904, 0.6009, 0.2566, 0.7936]])
        >>> generator = get_torch_generator(42)
        >>> torch.rand(2, 4, generator=generator)
        tensor([[0.8823, 0.9150, 0.3829, 0.9593],
                [0.3904, 0.6009, 0.2566, 0.7936]])
    """
    generator = torch.Generator(device)
    generator.manual_seed(random_seed)
    return generator


class BaseRandomSeedSetter(ABC):
    r"""Define the base class to implement a random seed setter.

    Each child class must implement the method ``manual_seed``.
    """

    @abstractmethod
    def manual_seed(self, seed: int) -> None:
        r"""Sets the seed for generating random numbers.

        Args:
            seed (int): Specifies the desired seed.

        Example usage:

        .. code-block:: python

            >>> from gravitorch.utils.seed import BaseRandomSeedSetter
            >>> setter: BaseRandomSeedSetter = ...  # Initialize a random seed setter.
            >>> setter.manual_seed(42)
        """


class NumpyRandomSeedSetter(BaseRandomSeedSetter):
    r"""Implements a random seed setter for the library ``numpy``.

    The seed must be between ``0`` and ``2**32 - 1``, so a modulo
    operator to convert an integer to an integer between ``0`` and
    ``2**32 - 1``.
    """

    def __str__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def manual_seed(self, seed: int) -> None:
        numpy.random.seed(seed % 2**32)


class RandomRandomSeedSetter(BaseRandomSeedSetter):
    r"""Implements a random seed setter for the python standard library
    ``random``."""

    def __str__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def manual_seed(self, seed: int) -> None:
        random.seed(seed)


class TorchRandomSeedSetter(BaseRandomSeedSetter):
    r"""Implements a random seed setter for the library ``torch``."""

    def __str__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def manual_seed(self, seed: int) -> None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


class RandomSeedSetter(BaseRandomSeedSetter):
    r"""Implements the default random seed setter.

    By default, it is initialized with the following random seed setters:

        - ``'numpy'``: ``NumpyRandomSeedSetter``
        - ``'random'``: ``RandomRandomSeedSetter``
        - ``'torch'``: ``TorchRandomSeedSetter``
    """

    registry: dict[str, BaseRandomSeedSetter] = {
        "numpy": NumpyRandomSeedSetter(),
        "random": RandomRandomSeedSetter(),
        "torch": TorchRandomSeedSetter(),
    }

    def __repr__(self) -> str:
        setters = {key: value for key, value in self.registry.items()}
        return f"{self.__class__.__qualname__}(\n" f"  {str_indent(to_pretty_dict_str(setters))}\n)"

    def manual_seed(self, seed: int) -> None:
        for value in self.registry.values():
            value.manual_seed(seed)

    @classmethod
    def add_setter(cls, name: str, setter: "BaseRandomSeedSetter", exist_ok: bool = False) -> None:
        r"""Adds a random seed setter for a given name.

        Args:
            name (str): Specifies the name for the setter.
            setter (``BaseRandomSeedSetter``): Specifies the random
                seed setter to add.
            exist_ok (bool, optional): If ``False``, ``ValueError`` is
                raised if the name already exists. This parameter
                should be set to ``True`` to overwrite the setter for
                a name. Default: ``False``.

        Raises:
            ValueError if a random seed setter is already registered
                for the name and ``exist_ok=False``.

        Example usage:

        .. code-block:: python

            >>> from gravitorch.utils.seed import BaseRandomSeedSetter, RandomSeedSetter
            >>> class OtherRandomSeedSetter(BaseRandomSeedSetter):
            ...    def manual_seed(self, seed: int) -> None:
            ...        ...  # Custom implementation
            >>> RandomSeedSetter.add_setter('other', OtherRandomSeedSetter())
            # To overwrite an existing random seed setter
            >>> RandomSeedSetter.add_setter('other', OtherRandomSeedSetter(), exist_ok=True)
        """
        if name in cls.registry and not exist_ok:
            raise ValueError(
                f"A setter ({cls.registry[name]}) is already registered for the name {name}. "
                "Please use `exist_ok=True` if you want to overwrite the setter for this name"
            )
        cls.registry[name] = setter

    @classmethod
    def has_setter(cls, name: str) -> bool:
        r"""Indicates if a random seed setter is registered for the given name.

        Args:
            name (str): Specifies the name to check.

        Returns:
            bool: ``True`` if a random seed setter is registered,
                otherwise ``False``.

        Example usage:

        .. code-block:: python

            >>> from gravitorch.utils.seed import RandomSeedSetter
            >>> RandomSeedSetter.has_setter('torch')
            True
            >>> RandomSeedSetter.has_setter('missing')
            False
        """
        return name in cls.registry


def manual_seed(seed: int, setter: Optional[BaseRandomSeedSetter] = None) -> None:
    r"""Sets the seed for generating random numbers.

    Args:
        seed (int): Specifies the desired seed.
        setter (``BaseRandomSeedSetter`` or ``None``, optional):
            Specifies the random seed setters. If ``None``, the
            default random seed setter (``RandomSeedSetter``) is used.
            Default: ``None``

    Example usage:

    .. code-block:: python

        >>> from gravitorch.utils.seed import manual_seed
        >>> manual_seed(42)
        >>> torch.randn(3)
        tensor([0.3367, 0.1288, 0.2345])
        >>> torch.randn(3)
        tensor([ 0.2303, -1.1229, -0.1863])
        >>> manual_seed(42)
        >>> torch.randn(3)
        tensor([0.3367, 0.1288, 0.2345])

        # Set the seed only for numpy
        >>> from gravitorch.utils.seed import NumpyRandomSeedSetter
        >>> manual_seed(42, NumpyRandomSeedSetter())
        >>> torch.randn(3)
        tensor([ 2.2082, -0.6380,  0.4617])
    """
    setter = setter or RandomSeedSetter()
    setter.manual_seed(seed)


@contextmanager
def numpy_seed(seed: int):
    r"""Implements a context manager to manage the NumPy random seed and random
    number generator (RNG) state.

    The context manager sets the specified NumPy random seed and
    restores the original RNG state afterward.

    Args:
        seed (int): Specifies the random number generator seed to use
            while using this context manager.

    Example usage:

    .. code-block:: python

        >>> import numpy
        >>> from gravitorch.utils.seed import numpy_seed
        >>> with numpy_seed(42):
        ...     print(numpy.random.randn(2, 4))
        [[ 0.49671415 -0.1382643   0.64768854  1.52302986]
         [-0.23415337 -0.23413696  1.57921282  0.76743473]]
        >>> with numpy_seed(42):
        ...     print(numpy.random.randn((2, 4)))
        [[ 0.49671415 -0.1382643   0.64768854  1.52302986]
         [-0.23415337 -0.23413696  1.57921282  0.76743473]]
    """
    state = numpy.random.get_state()
    try:
        NumpyRandomSeedSetter().manual_seed(seed)
        yield
    finally:
        numpy.random.set_state(state)


@contextmanager
def torch_seed(seed: int):
    r"""Implements a context manager to manage the PyTorch random seed and
    random number generator (RNG) state.

    The context manager sets the specified PyTorch random seed and
    restores the original RNG state afterward.

    Args:
        seed (int): Specifies the random number generator seed to use
            while using this context manager.

    Example usage:

    .. code-block:: python

        >>> import torch
        >>> from gravitorch.utils.seed import torch_seed
        >>> with torch_seed(42):
        ...     print(torch.randn(2, 4))
        tensor([[ 0.3367,  0.1288,  0.2345,  0.2303],
                [-1.1229, -0.1863,  2.2082, -0.6380]])
        >>> with torch_seed(42):
        ...     print(torch.randn(2, 4))
        tensor([[ 0.3367,  0.1288,  0.2345,  0.2303],
                [-1.1229, -0.1863,  2.2082, -0.6380]])
    """
    state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all()
    try:
        TorchRandomSeedSetter().manual_seed(seed)
        yield
    finally:
        torch.set_rng_state(state)
        torch.cuda.set_rng_state_all(cuda_states)
