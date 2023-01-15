r"""This module implements some utility functions to use ``torch.Tensor``s."""

__all__ = [
    "has_name",
    "partial_transpose_dict",
    "permute",
    "shapes_are_equal",
    "str_full_tensor",
    "to_tensor",
]

from collections.abc import Sequence
from typing import Any, TypeVar

import numpy as np
import torch
from torch import Tensor

T = TypeVar("T")


def str_full_tensor(tensor: Tensor) -> str:
    r"""Computes a string representation of the tensor with all the values.

    Args:
        tensor (``torch.Tensor``): Specifies the input tensor.

    Returns:
         str: The string representation of the tensor with all the
            values.
    """
    torch.set_printoptions(profile="full")
    output = str(tensor)
    torch.set_printoptions(profile="default")  # reset
    return output


def has_name(tensor: Tensor) -> bool:
    r"""Indicates it the tensor has at least one name or not.

    Args:
        tensor (``torch.Tensor``): Specifies the tensor to check.

    Returns:
        bool: ``True`` if tensor has at least one name, otherwise
            ``False``.

    Example usage:

    .. code-block:: python

        >>> from gravitorch.utils.tensor import has_name
        >>> has_name(torch.ones(2, 3))
        False
        >>> has_name(torch.ones(2, 3, names=('B', None)))
        True
        >>> has_name(torch.ones(2, 3, names=('B', 'F')))
        True
    """
    return set(tensor.names) != {None}


def permute(tensor: Tensor, permutation: Tensor, dim: int = 0) -> Tensor:
    r"""Permutes a tensor on a given dimension and a permutation.

    Args:
        tensor (``torch.Tensor``): Specifies the tensor to permute.
        permutation (``torch.Tensor`` of type long and shape
            ``(dimension,)``): Specifies the permutation to use on the
            tensor. The dimension of this tensor should be compatible
            with the shape of the tensor to permute.
        dim (int, optional): Specifies the dimension used to permute the
            tensor. Default: ``0``

    Returns:
        ``torch.Tensor``: The permuted tensor.

    Example usage:

    .. code-block:: python

        >>> from gravitorch.utils.tensor import permute
        >>> permute(tensor=torch.arange(4), permutation=torch.tensor([0, 2, 1, 3]))
        tensor([0, 2, 1, 3])
        >>> permute(tensor=torch.arange(20).view(4, 5), permutation=torch.tensor([0, 2, 1, 3]))
        tensor([[0, 1, 2, 3, 4], [10, 11, 12, 13, 14], [5, 6, 7, 8, 9], [15, 16, 17, 18, 19]])
        >>> permute(
        ...     tensor=torch.arange(20).view(4, 5),
        ...     permutation=torch.tensor([0, 4, 2, 1, 3]),
        ...     dim=1,
        ... )
        tensor([[ 0,  4,  2,  1,  3],
                [ 5,  9,  7,  6,  8],
                [10, 14, 12, 11, 13],
                [15, 19, 17, 16, 18]])
        >>> permute(
        ...     tensor=torch.arange(20).view(2, 2, 5),
        ...     permutation=torch.tensor([0, 4, 2, 1, 3]),
        ...     dim=2,
        ... )
        tensor([[[ 0,  4,  2,  1,  3],
                 [ 5,  9,  7,  6,  8]],
                [[10, 14, 12, 11, 13],
                 [15, 19, 17, 16, 18]]])
    """
    return tensor.transpose(0, dim)[permutation].transpose(0, dim).contiguous()


def partial_transpose_dict(data: dict, config: dict) -> dict:
    r"""Transposes some ``torch.Tensor``s that are in a dictionary.

    Note the transposed tensors may not be contiguous in memory
    tensors.

    Args:
        data (dict): Specifies the dictionary with the tensors to
            transpose.
        config (dict): Specifies the tensors to transpose. The keys
            should exist in the ``data`` input. The keys indicate the
            tensors to transpose, and the values indicate the
            dimension to transpose. See example below for more
            details.

    Returns:
        dict: A dictionary with some transposed tensors.

    Example usage:

    .. code-block:: python

        >>> x = torch.arange(10).view(2, 5)
        >>> x
        tensor([[0, 1, 2, 3, 4],
                [5, 6, 7, 8, 9]])
        # No transposition
        >>> from gravitorch.utils.tensor import partial_transpose_dict
        >>> partial_transpose_dict({'my_key': x}, {})
        {'my_key': tensor([[0, 1, 2, 3, 4],
                 [5, 6, 7, 8, 9]])}
        # Transpose the first two dimensions
        >>> partial_transpose_dict({'my_key': x}, {'my_key': [0, 1]})
        {'my_key': tensor([[0, 5],
                 [1, 6],
                 [2, 7],
                 [3, 8],
                 [4, 9]])}
        # The order of the dimensions is not important  [0, 1] <> [1, 0]
        >>> partial_transpose_dict({'my_key': x}, {'my_key': [1, 0]})
        {'my_key': tensor([[0, 5],
                 [1, 6],
                 [2, 7],
                 [3, 8],
                 [4, 9]])}
        >>> x = torch.arange(24).view(2, 3, 4)
        >>> x
        tensor([[[ 0,  1,  2,  3],
                 [ 4,  5,  6,  7],
                 [ 8,  9, 10, 11]],

                [[12, 13, 14, 15],
                 [16, 17, 18, 19],
                 [20, 21, 22, 23]]])
        >>> partial_transpose_dict({'my_key': x}, {'my_key': [0, 1]})
        {'my_key': tensor([[[ 0,  1,  2,  3],
                  [12, 13, 14, 15]],

                 [[ 4,  5,  6,  7],
                  [16, 17, 18, 19]],

                 [[ 8,  9, 10, 11],
                  [20, 21, 22, 23]]])}
        >>> partial_transpose_dict({'my_key': x}, {'my_key': [0, 2]})
        {'my_key': tensor([[[ 0, 12],
                  [ 4, 16],
                  [ 8, 20]],

                 [[ 1, 13],
                  [ 5, 17],
                  [ 9, 21]],

                 [[ 2, 14],
                  [ 6, 18],
                  [10, 22]],

                 [[ 3, 15],
                  [ 7, 19],
                  [11, 23]]])}
        >>> partial_transpose_dict({'my_key': x}, {'my_key': [1, 2]})
        {'my_key': tensor([[[ 0,  4,  8],
                  [ 1,  5,  9],
                  [ 2,  6, 10],
                  [ 3,  7, 11]],

                 [[12, 16, 20],
                  [13, 17, 21],
                  [14, 18, 22],
                  [15, 19, 23]]])}
    """
    for key, dims in config.items():
        if key not in data:
            raise ValueError(f"key '{key}' is not in the data dict (keys: {list(data.keys())})")
        data[key] = data[key].transpose(*dims)
    return data


def to_tensor(value: Any) -> Tensor:
    r"""Converts the input to a ``torch.Tensor``.

    Args:
        value: Specifies the input to convert to ``torch.Tensor``.

    Returns:
        ``torch.Tensor``: The tensor.

    Raises:
        TypeError if the type cannot be converted to ``torch.Tensor``.

    Example usage:

    .. code-block:: python

        >>> import torch
        >>> from gravitorch.utils.tensor import to_tensor
        >>> to_tensor(torch.tensor([-3, 1, 7]))
        tensor([-3,  1,  7])
        >>> to_tensor([-3, 1, 7])
        tensor([-3,  1,  7])
        >>> to_tensor((-3, 1, 7))
        tensor([-3,  1,  7])
        >>> to_tensor(np.array([-3, 1, 7]))
        tensor([-3,  1,  7])
    """
    if torch.is_tensor(value):
        return value
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    if isinstance(value, Sequence):
        return torch.tensor(value)
    if isinstance(value, (int, float)):
        return torch.tensor(value)
    raise TypeError(f"Incorrect type: {type(value)}")


def shapes_are_equal(tensors: Sequence[Tensor]) -> bool:
    r"""Indicates if the shapes of several tensors are equal or not.

    This method does not check the values or the data type of the
    tensors.

    Args:
        tensors (sequence): Specifies the tensors to check.

    Returns:
        bool: ``True`` if all the tensors have the same shape,
            otherwise ``False``. By design, this function returns
            ``False`` if no tensor is provided.

    Example usage:

    .. code-block:: python

        >>> import torch
        >>> from gravitorch.utils.tensor import shapes_are_equal
        >>> shapes_are_equal([torch.rand(2, 3), torch.rand(2, 3)])
        True
        >>> shapes_are_equal([torch.rand(2, 3), torch.rand(2, 3, 1)])
        False
    """
    if not tensors:
        return False
    shape = tensors[0].shape
    for tensor in tensors[1:]:
        if shape != tensor.shape:
            return False
    return True
