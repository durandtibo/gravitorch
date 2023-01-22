r"""This module implements some utility functions to manipulate
mappings/dicts."""

__all__ = ["convert_to_dict_of_lists", "convert_to_list_of_dicts", "get_first_value"]

from collections.abc import Hashable, Mapping, Sequence
from typing import Any


def convert_to_dict_of_lists(seq_of_mappings: Sequence[Mapping]) -> dict[Hashable, list]:
    r"""Convert a sequence of mappings to a dictionary of lists.

    All the dictionaries should have the same keys. The first
    dictionary in the sequence is used to find the keys.

    Args:
        seq_of_mappings (sequence): Specifies the sequence of
            mappings.

    Returns:
        dict: A dictionary of lists.

    Example usage:

    .. code-block:: python

        >>> from gravitorch.utils.mapping import convert_to_dict_of_lists
        >>> convert_to_dict_of_lists(
        ...     [{"key1": 1, "key2": 10}, {"key1": 2, "key2": 20}, {"key1": 3, "key2": 30}]
        ... )
        {'key1': [1, 2, 3], 'key2': [10, 20, 30]}
    """
    if seq_of_mappings:
        return {key: [dic[key] for dic in seq_of_mappings] for key in seq_of_mappings[0]}
    return {}


def convert_to_list_of_dicts(mapping_of_seqs: Mapping[Hashable, Sequence]) -> list[dict]:
    r"""Convert a mapping of sequences to a list of dictionaries.

    All the sequences should have the same length.

    Args:
        mapping_of_seqs (``Mapping``): Specifies a mapping of
            sequences.

    Returns:
        dict: A dictionary of lists.

    Example usage:

    .. code-block:: python

        >>> from gravitorch.utils.mapping import convert_to_list_of_dicts
        >>> convert_to_list_of_dicts({"key1": [1, 2, 3], "key2": [10, 20, 30]})
        [{'key1': 1, 'key2': 10}, {'key1': 2, 'key2': 20}, {'key1': 3, 'key2': 30}]
    """
    return [dict(zip(mapping_of_seqs, seqs)) for seqs in zip(*mapping_of_seqs.values())]


def remove_keys_starting_with(mapping: Mapping, prefix: str) -> dict:
    r"""Removes the keys that start with a given prefix.

    Args:
        mapping (``Mapping``): Specifies the original mapping.

    Returns:
        dict: A new dict without the removed keys.

    Example usage:

    .. code-block:: python

        >>> from gravitorch.utils.mapping import remove_keys_starting_with
        >>> remove_keys_starting_with(
        ...     {"key": 1, "key.abc": 2, "abc": 3, "abc.key": 4, 1: 5, (2, 3): 6}, "key",
        ... )
        {'abc': 3, 'abc.key': 4, 1: 5, (2, 3): 6}
    """
    new_dict = {}
    for key, value in mapping.items():
        if isinstance(key, str) and key.startswith(prefix):
            continue
        new_dict[key] = value
    return new_dict


def get_first_value(data: Mapping) -> Any:
    r"""Gets the first value of a mapping.

    Args:
        data (``Mapping``): Specifies the input mapping.

    Returns:
        The first value in the mapping.

    Example usage:

    .. code-block:: python

        >>> from gravitorch.utils.mapping import get_first_value
        >>> get_first_value({"key1": 1, "key2": 2})
        1
    """
    if not data:
        raise ValueError("First value cannot be returned because the mapping is empty")
    return data[next(iter(data))]
