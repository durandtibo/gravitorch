__all__ = [
    "TensorDictShufflerIterDataPipe",
    "shuffle_tensors",
    "shuffle_tensor_mapping",
    "get_first_dimension",
]

import logging
from collections.abc import Iterator, Mapping, Sequence
from typing import Optional, Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import IterDataPipe

from gravitorch.utils.format import str_indent
from gravitorch.utils.seed import get_torch_generator
from gravitorch.utils.tensor import permute

logger = logging.getLogger(__name__)


class TensorDictShufflerIterDataPipe(IterDataPipe[dict]):
    r"""Implements a source DataPipe to shuffle ``torch.Tensor``s in
    dictionaries.

    For each dictionary, the ``torch.Tensor``s are shuffled with the
    same permutation on a given dimension. This function shuffles
    each tensor on one dimension. It is possible to control the
    dimension used to shuffle the tensors with the argument ``dim``.
    There are two approaches:

        - If ``dim`` is an integer, all the tensors are shuffled on the
            dimension indicated by ``dim``. All the tensors should have
            the same dimension for the specified dimension
            ``(..., dimension, *)`` where ``*`` means any number of
            dimensions.
        - If ``dim`` is a dictionary, each tensor can be shuffled on a
            specific dimension. ``dim`` should indicate the dimension
            to use for each tensor. The keys in this dictionary should
            include keys that are in the input ``mapping``. Note that
            ``dim`` can contain only a subset of keys in the input
            ``mapping``. See example in ``shuffle_tensor_mapping``.

    Args:
        datapipe (``IterDataPipe``): Specifies the source
            iterable DataPipe.
        dim (int or dict, optional): Specifies the dimension used to
            shuffle the mapping. Default: ``0``
        random_seed (int, optional): Specifies the random seed used
            to shuffle the ``torch.Tensor``s.
            Default: ``3510637111256283951``

    Example usage:

    .. code-block:: python

        >>> from torch.utils.data import IterDataPipe
        >>> from gravitorch.data.datapipes.iter import TensorDictShuffler
        >>> class MyIterDataPipe(IterDataPipe[dict]):
        ...     def __iter__(self) -> Iterator[dict]:
        ...         for i in range(3):
        ...             yield {"key": torch.arange(4) + i}
        ...
        >>> list(TensorDictShuffler(MyIterDataPipe()))
        [{'key': tensor([2, 3, 0, 1])},
         {'key': tensor([3, 2, 4, 1])},
         {'key': tensor([2, 5, 4, 3])}]
        >>> list(TensorDictShuffler(MyIterDataPipe(), dim={'key': 0}))
        [{'key': tensor([2, 3, 0, 1])},
         {'key': tensor([3, 2, 4, 1])},
         {'key': tensor([2, 5, 4, 3])}]
    """

    def __init__(
        self,
        datapipe: IterDataPipe[Mapping],
        dim: Union[int, dict] = 0,
        random_seed: int = 3510637111256283951,
    ):
        self._datapipe = datapipe
        self._dim = dim
        self._generator = get_torch_generator(random_seed)

    def __iter__(self) -> Iterator[dict]:
        for data in self._datapipe:
            yield shuffle_tensor_mapping(data, dim=self._dim, generator=self._generator)

    def __len__(self) -> int:
        return len(self._datapipe)

    def __str__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  dim={self._dim},\n"
            f"  random_seed={self.random_seed},\n"
            f"  datapipe={str_indent(self._datapipe)},\n)"
        )

    @property
    def random_seed(self) -> int:
        r"""int: The random seed used to initialize the pseudo random generator."""
        return self._generator.initial_seed()


def shuffle_tensors(
    tensors: Sequence[Tensor],
    dim: int = 0,
    generator: Optional[torch.Generator] = None,
) -> list[Tensor]:
    r"""Shuffles tensors with the same permutation on a given dimension.

    This function shuffles the tensors only on a dimension. It is
    possible to control the dimension used to shuffle the tensors with
    the argument ``dim``. All the tensors should have the same
    dimension for the specified dimension ``(..., dimension, *)`` where
    ``*`` means any number of dimensions.

    Args:
        tensors (``Sequence``): Specifies the tensors to shuffle.
        dim (int, optional): Specifies the dimension used to shuffle
            the tensors. Default: ``0``
        generator (``torch.Generator`` or ``None``, optional):
            Specifies an optional random generator. Default: ``None``

    Returns:
        ``list``: The shuffled tensors.

    Raises:
        ValueError if the tensors do not have the same shape for the
            common dimension.

    Example usage:

    .. code-block:: python

        >>> from gravitorch.data.datapipes.iter.shuffling import shuffle_tensors
        >>> shuffle_tensors([torch.arange(4), torch.arange(20).view(4, 5)])
        [
            torch.tensor([0, 2, 1, 3]),
            torch.tensor([[0, 1, 2, 3, 4],
                          [10, 11, 12, 13, 14],
                          [5, 6, 7, 8, 9],
                          [15, 16, 17, 18, 19]]),
        ]
        >>> shuffle_tensors([torch.arange(4).view(1, 4), torch.arange(20).view(5, 4)], dim=1)
        [
            torch.tensor([[0, 2, 1, 3]]),
            torch.tensor([[0, 2, 1, 3],
                          [4, 6, 5, 7],
                          [8, 10, 9, 11],
                          [12, 14, 13, 15],
                          [16, 18, 17, 19]]),
        ]
    """
    dims = [tensor.shape[dim] for tensor in tensors]
    if len(set(dims)) > 1:
        raise ValueError(
            f"The tensors do not have the same shape for the common dimension (dim={dim}): {dims}"
        )
    permutation = torch.randperm(tensors[0].shape[dim], generator=generator)
    return [permute(tensor, permutation, dim) for tensor in tensors]


def shuffle_tensor_mapping(
    mapping: Mapping, dim: Union[int, dict] = 0, generator: Optional[torch.Generator] = None
) -> dict:
    r"""Shuffles the tensors that are in a mapping with the same permutation.

    This function shuffles each tensor on one dimension. It is
    possible to control the dimension used to shuffle the tensors
    with the argument ``dim``. There are two approaches:

        - If ``dim`` is an integer, all the tensors are shuffled on the
            dimension indicated by ``dim``. All the tensors should have
            the same dimension for the specified dimension
            ``(..., dimension, *)`` where ``*`` means any number of
            dimensions.
        - If ``dim`` is a dictionary, each tensor can be shuffled on a
            specific dimension. ``dim`` should indicate the dimension
            to use for each tensor. The keys in this dictionary should
            include keys that are in the input ``mapping``. Note that
            ``dim`` can contain only a subset of keys in the input
            ``mapping``. See example below.

    Args:
        mapping (``Mapping``): Specifies the mapping with the tensors
            to shuffle.
        dim (int or dict, optional): Specifies the dimension used to
            shuffle the mapping. Default: ``0``
        generator (``torch.Generator`` or ``None``, optional):
            Specifies an optional random generator. Default: ``None``

    Returns:
        ``dict``: A dictionary with the shuffled tensors.

    Raises:
        ValueError if the tensors do not have the same shape for the
            common dimension.

    Example usage:

    .. code-block:: python

        >>> from gravitorch.data.datapipes.iter.shuffling import shuffle_tensor_mapping
        # Shuffle all the tensors on dimension 0 (default)
        >>> shuffle_tensor_mapping({"key1": torch.arange(4), "key2": torch.arange(20).view(4, 5)})
        {
            "key1": tensor([0, 2, 1, 3]),
            "key2": tensor([[0, 1, 2, 3, 4],
                            [10, 11, 12, 13, 14],
                            [5, 6, 7, 8, 9],
                            [15, 16, 17, 18, 19]]),
        }
        # Shuffle all the tensors on dimension 1
        >>> shuffle_tensor_mapping(
        ...     {"key1": torch.arange(4).view(1, 4), "key2": torch.arange(20).view(5, 4)},
        ...     dim=1,
        ... )
        {
            "key1": tensor([[0, 2, 1, 3]]),
            "key2": tensor([[0, 2, 1, 3],
                            [4, 6, 5, 7],
                            [8, 10, 9, 11],
                            [12, 14, 13, 15],
                            [16, 18, 17, 19]]),
        }
        # Shuffle the tensor "key1" on dimension 0 and the tensor "key2" on dimension 1
        >>> shuffle_tensor_mapping(
        ...     {"key1": torch.arange(4), "key2": torch.arange(20).view(5, 4)},
        ...     dim={'key1': 0, 'key2': 1}
        ... )
        {
            'key1': tensor([1, 2, 0, 3]),
            'key2': tensor([[1,  2,  0,  3],
                            [5,  6,  4,  7],
                            [9, 10,  8, 11],
                            [13, 14, 12, 15],
                            [17, 18, 16, 19]]),
        }
        # Shuffle only the tensor "key2" on dimension 1
        >>> shuffle_tensor_mapping(
        ...     {"key1": torch.arange(4), "key2": torch.arange(20).view(5, 4)},
        ...     dim={'key2': 1},
        ... )
        {
            'key1': tensor([0, 1, 2, 3]),
            'key2': tensor([[3,  1,  0,  2],
                            [7,  5,  4,  6],
                            [11,  9,  8, 10],
                            [15, 13, 12, 14],
                            [19, 17, 16, 18]]),
        }
    """
    dims = dim if isinstance(dim, dict) else {key: dim for key in mapping.keys()}
    if len(dims) == 0:  # No tensor to permute
        return {key: tensor for key, tensor in mapping.items()}
        # Check if the common dimensions are the same
    valid_dims = [mapping[key].shape[dims[key]] for key in mapping.keys() if key in dims]
    if len(set(valid_dims)) > 1:
        raise ValueError(
            f"The tensors do not have the same shape for the common dimension: {valid_dims}"
        )
    # Permute the tensors
    key = next(iter(dims.keys()))
    permutation = torch.randperm(mapping[key].shape[dims[key]], generator=generator)
    output = {}
    for key, tensor in mapping.items():
        if key in dims:
            tensor = permute(tensor, permutation, dims[key])
        output[key] = tensor
    return output


def get_first_dimension(data: Union[Tensor, np.ndarray, list, tuple]) -> int:
    r"""Finds the first dimension value.

    This function finds the first dimension value by using the
    following rules:

        - if the input is a ``torch.Tensor`` or ``numpy.ndarray``,
            the first dimension value is returned.
        - if the input is a list or tuple, the length is returned.

    Args:
        data: Specifies the data to get the first dimension value.

    Returns:
        int: The first dimension value.

    Raises:
        TypeError: if the input type is not supported.

    Example usage:

    .. code-block:: python

        >>> from gravitorch.data.datapipes.iter.shuffling import get_first_dimension
        # torch tensor
        >>> import torch
        >>> get_first_dimension(torch.ones(5, 4))
        5
        # NumPy ndarray
        >>> import numpy
        >>> get_first_dimension(numpy.ones((7, 4)))
        7
        # List/tuple
        >>> get_first_dimension([1, 3, 4])
        3
        >>> get_first_dimension((5, 2, 1, 4))
        4
    """
    if torch.is_tensor(data) or isinstance(data, np.ndarray):
        return data.shape[0]
    if isinstance(data, (list, tuple)):
        return len(data)
    raise TypeError(
        f"Incorrect type. The supported types are: torch.Tensor, numpy.ndarray, list and tuple. "
        f"(received: {data})"
    )
