r"""This module implements some utility functions to do standard element-wise
operations on some data structures."""

__all__ = ["add_objects", "sub_objects"]

from typing import Any


def add_objects(object1: Any, object2: Any) -> Any:
    r"""Performs an element-wise addition of two objects.

    Equivalent to: ``object1 + object2``.

    The current implementation supports the following object types:

        - int
        - float
        - ``torch.Tensor``
        - ``numpy.ndarray``
        - ``list``
        - ``tuple``
        - ``dict``

    The set are not supported because it is not always possible to do
    an element-wise addition. This function assumes that both objects
    have the same structure, and it is possible to do an element-wise
    operation between the values. For example if
    ``object1['my_list']`` is a list, it is expected that
    ``object2['my_list']`` is also a list.

    Args:
        object1: Specifies the first object
        object2: Specifies the second object

    Returns:
        The element-wise addition of the two objects.

    Example usage:

    .. code-block:: python

        >>> import torch
        >>> from gravitorch.experimental.object_ops import add_objects
        >>> add_objects([1, 2, 3], [4, 5, 6])
        [5, 7, 9]
        >>> add_objects(
        ...     (torch.tensor([1, 2]), torch.tensor([3])),
        ...     (torch.tensor([4, 5]), torch.tensor([6])),
        ... )
        (torch.tensor([5, 7]), torch.tensor([9]))
    """
    if isinstance(object1, dict):
        return type(object1)(
            {key: add_objects(value, object2[key]) for key, value in object1.items()}
        )
    if isinstance(object1, (tuple, list)):
        return type(object1)(add_objects(obj1, obj2) for obj1, obj2 in zip(object1, object2))
    return object1 + object2


def sub_objects(object1: Any, object2: Any) -> Any:
    r"""Performs an element-wise substraction of two objects.

    Equivalent to: ``object1 - object2``

    The current implementation supports the following object types:

        - int
        - float
        - ``torch.Tensor``
        - ``numpy.ndarray``
        - ``list``
        - ``tuple``
        - ``dict``

    The set are not supported because it is not always possible to do
    an element-wise addition. This function assumes that both objects
    have the same structure, and it is possible to do an element-wise
    operation between the values. For example if
    ``object1['my_list']`` is a list, it is expected that
    ``object2['my_list']`` is also a list.

    Args:
        object1: Specifies the first object
        object2: Specifies the second object

    Returns:
        The element-wise substraction of the two objects.

    Example usage:

    .. code-block:: python

        >>> import torch
        >>> from gravitorch.experimental.object_ops import sub_objects
        >>> sub_objects([1, 2, 3], [4, 5, 6])
        [-3, -3, -3]
        >>> sub_objects(
        ...     (torch.tensor([4, 5]), torch.tensor([6])),
        ...     (torch.tensor([1, 0]), torch.tensor([3])),
        ... )
        (torch.tensor([3, 5]), torch.tensor([3]))
    """
    if isinstance(object1, dict):
        return type(object1)(
            {key: sub_objects(value, object2[key]) for key, value in object1.items()}
        )
    if isinstance(object1, (tuple, list)):
        return type(object1)(sub_objects(obj1, obj2) for obj1, obj2 in zip(object1, object2))
    return object1 - object2
