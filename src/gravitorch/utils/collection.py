from __future__ import annotations

__all__ = ["to_tuple"]

from typing import Any


def to_list(value: Any) -> list:
    r"""Converts a value to a list.

    This function is a no-op if the input is a list.

    Args:
    ----
        value: Specifies the value to convert.

    Returns:
    -------
        list: The input value in a list.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.utils import to_list
        >>> to_list(1)
        [1]
        >>> to_list("abc")
        ['abc']
    """
    if isinstance(value, list):
        return value
    if isinstance(value, (bool, int, float, str)):
        return [value]
    return list(value)


def to_tuple(value: Any) -> tuple:
    r"""Converts a value to a tuple.

    This function is a no-op if the input is a tuple.

    Args:
    ----
        value: Specifies the value to convert.

    Returns:
    -------
        tuple: The input value in a tuple.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.utils import to_tuple
        >>> to_tuple(1)
        (1,)
        >>> to_tuple("abc")
        ('abc',)
    """
    if isinstance(value, tuple):
        return value
    if isinstance(value, (bool, int, float, str)):
        return (value,)
    return tuple(value)
