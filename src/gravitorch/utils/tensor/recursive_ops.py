r"""This module implements some utility functions to use ``torch.Tensor``s."""

__all__ = [
    "UNKNOWN",
    "get_dtype",
    "get_shape",
    "recursive_apply",
    "recursive_contiguous",
    "recursive_detach",
    "recursive_transpose",
]

from collections.abc import Callable
from typing import Any, Optional, TypeVar

import numpy as np
import torch

T = TypeVar("T")

UNKNOWN = "unknown"


def get_dtype(data: Any) -> Any:
    r"""Gets the tensor data type recursively in a data structure.

    The current implementation supports the following types:

        - ``collections.OrderedDict``
        - ``dict``
        - ``list``
        - ``torch.Tensor``
        - ``tuple``

    Args:
        data: Specifies the input data.

    Returns:
        An object with the same structure as the input but with the
            tensor data type.

    Example usage:

    .. code-block:: python

        >>> from gravitorch.utils.tensor import get_dtype
        >>> get_dtype(torch.ones(1, 2, 3))
        torch.Size([1, 2, 3])
        >>> get_dtype({"key1": torch.ones(1, 2, 3), "key2": "abc"})
        {'key1': torch.Size([1, 2, 3]), 'key2': 'unknown'}
        >>> get_dtype([torch.ones(1, 2, 3), "abc"])
        [torch.Size([1, 2, 3]), 'unknown']
    """
    if torch.is_tensor(data):
        return data.dtype
    if isinstance(data, (list, tuple, set)):
        return type(data)(get_dtype(t) for t in data)
    if isinstance(data, dict):
        return type(data)({key: get_dtype(value) for key, value in data.items()})
    return UNKNOWN


def get_shape(data: Any) -> Any:
    r"""Gets the tensor shape recursively in a data structure.

    The current implementation supports the following types:

        - ``collections.OrderedDict``
        - ``dict``
        - ``list``
        - ``torch.Tensor``
        - ``tuple``

    Args:
        data: Specifies the input data.

    Returns:
        An object with the same structure as the input but with the
            tensor shapes.

    Example usage:

    .. code-block:: python

        >>> from gravitorch.utils.tensor import get_shape
        >>> get_shape(torch.ones(1, 2, 3))
        torch.Size([1, 2, 3])
        >>> get_shape({"key1": torch.ones(1, 2, 3), "key2": "abc"})
        {'key1': torch.Size([1, 2, 3]), 'key2': 'unknown'}
        >>> get_shape([torch.ones(1, 2, 3), "abc"])
        [torch.Size([1, 2, 3]), 'unknown']
    """
    if torch.is_tensor(data):
        return data.shape
    if isinstance(data, (list, tuple, set)):
        return type(data)(get_shape(t) for t in data)
    if isinstance(data, dict):
        return type(data)({key: get_shape(value) for key, value in data.items()})
    return UNKNOWN


def recursive_apply(data: Any, tensor_fn: Callable, other_fn: Optional[Callable] = None) -> Any:
    r"""Recursively applies a function on all the ``torch.Tensor``s.

    The current implementation supports the following types:

        - ``collections.OrderedDict``
        - ``dict``
        - ``list``
        - ``set``
        - ``torch.Tensor``
        - ``tuple``

    Args:
        data: Specifies the data.
        tensor_fn (callable): Specifies the function to apply on
            ``torch.Tensor``s.
        other_fn (callable or ``None``, optional): Specifies the
            function to apply on non ``torch.Tensor`` values.
            ``None`` means the original value is returned.
            Default: ``None``

    Returns:
        The transformed data.

    Example usage:

    .. code-block:: python

        >>> import torch
        >>> from gravitorch.utils.tensor import recursive_apply
        >>> recursive_apply(torch.ones(2, 3), lambda tensor: tensor.dtype)
        torch.float32
        >>> recursive_apply(
        ...     {
        ...         'key1': torch.ones(2, 3),
        ...         'key2': 'abc',
        ...         'key3': [torch.ones(2, 3), torch.arange(6)]
        ...     },
        ...     lambda tensor: tensor.dtype,
        ... )
        {'key1': torch.float32, 'key2': 'abc', 'key3': [torch.float32, torch.int64]}
    """
    if torch.is_tensor(data):
        return tensor_fn(data)
    if isinstance(data, (list, tuple, set)):
        return type(data)(recursive_apply(t, tensor_fn, other_fn) for t in data)
    if isinstance(data, dict):
        return type(data)(
            {key: recursive_apply(value, tensor_fn, other_fn) for key, value in data.items()}
        )
    return other_fn(data) if other_fn else data


def recursive_contiguous(
    data: T, memory_format: torch.memory_format = torch.contiguous_format
) -> T:
    r"""Returns contiguous in memory tensors containing the same data as the
    input.

    The current implementation supports the following types:

        - ``collections.OrderedDict``
        - ``dict``
        - ``list``
        - ``set``
        - ``torch.Tensor``
        - ``tuple``

    Args:
        data: Specifies the data to transform tocontiguous in memory
            tensors. All the tensors will be transformed by using the
            ``contiguous`` method.
        memory_format (``torch.memory_format``, optional): Specifies
            the desired memory format.
            Default: ``torch.contiguous_format``

    Returns:
        Contiguous in memory tensors containing the same data as the
            input.

    Example usage:

    .. code-block:: python

        >>> from gravitorch.utils.tensor import recursive_contiguous
        >>> x = torch.ones(2, 3).transpose(0, 1)
        >>> x.is_contiguous()
        False
        >>> recursive_contiguous(x).is_contiguous()
        True
        >>> out = recursive_contiguous({'key': x})
        >>> out['key'].is_contiguous()
        True
    """
    return recursive_apply(data, lambda tensor: tensor.contiguous(memory_format=memory_format))


def recursive_detach(data: T) -> T:
    r"""Detaches the ``torch.Tensor``s.

    The current implementation supports the following types:

        - ``collections.OrderedDict``
        - ``dict``
        - ``list``
        - ``set``
        - ``torch.Tensor``
        - ``tuple``

    Args:
        data: Specifies the data to detach. All the tensors will be
            detached by using the ``detach`` method.

    Returns:
        The detached data.

    Example usage:

    .. code-block:: python

        >>> from gravitorch.utils.tensor import recursive_detach
        >>> recursive_detach(torch.ones(2, 3, requires_grad=True)).requires_grad
        False
        >>> out = recursive_detach({'key': torch.ones(2, 3, requires_grad=True)})
        >>> out['key'].requires_grad
        False
    """
    return recursive_apply(data, lambda tensor: tensor.detach())


def recursive_transpose(data: T, dim0: int, dim1: int) -> T:
    r"""Transposes all the ``torch.Tensor``s in the input.

    The current implementation supports the following types:

        - ``collections.OrderedDict``
        - ``dict``
        - ``list``
        - ``set``
        - ``torch.Tensor``
        - ``tuple``

    Note: all the tensors should be compatible with the transpose
    dimensions.

    Args:
        data: Specifies the data to transpose. All the
            ``torch.Tensor``s are transposed by using the
            ``transpose`` method.
        dim0 (int): Specifies the first dimension to be transposed.
        dim1 (int): Specifies the second dimension to be transposed.

    Returns:
        The input data where the ``torch.Tensor``s are transposed.

    Example usage:

    .. code-block:: python

        >>> from gravitorch.utils.tensor import recursive_transpose
        >>> recursive_transpose(torch.ones(3, 2), 0, 1).shape
        torch.Size([2, 3])
        >>> out = recursive_transpose({'key': torch.ones(3, 2)})
        >>> out['key'].shape
        torch.Size([2, 3])
    """
    return recursive_apply(data, lambda tensor: tensor.transpose(dim0, dim1))


def recursive_from_numpy(data: Any) -> Any:
    r"""Comverts recursively all the ``numpy.ndarray``s to ``torch.Tensor``s.

    The current implementation supports the following types:

        - ``collections.OrderedDict``
        - ``dict``
        - ``list``
        - ``numpy.ndarray``
        - ``set``
        - ``torch.Tensor``
        - ``tuple``

    Args:
        data: Specifies the data to convert.

    Returns:
        The input data where all the ``numpy.ndarray``s are converted
            to ``torch.Tensor``s.

    Example usage:

    .. code-block:: python

        >>> import numpy as np
        >>> from gravitorch.utils.tensor import recursive_from_numpy
        >>> recursive_from_numpy(np.ones((3, 2)))
        tensor([[1., 1.],
                [1., 1.],
                [1., 1.]], dtype=torch.float64)
        >>> recursive_from_numpy({'key': np.ones((3, 2), dtype=np.float32)})
        {'key': tensor([[1., 1.],
                [1., 1.],
                [1., 1.]])}
    """
    return recursive_apply(
        data,
        lambda tensor: tensor,
        lambda value: torch.from_numpy(value) if isinstance(value, np.ndarray) else value,
    )
