r"""This module defines some utility functions to format some
objects."""

from __future__ import annotations

__all__ = [
    "human_byte_size",
    "human_count",
    "human_time",
    "str_mapping",
    "str_pretty_dict",
    "str_pretty_json",
    "str_pretty_yaml",
    "str_scalar",
    "str_target_object",
]

import datetime
import json
import math
from collections.abc import Mapping
from typing import Any, TypeVar

import yaml
from coola.utils import str_indent
from objectory import OBJECT_TARGET

PARAMETER_NUM_UNITS = (" ", "K", "M", "B", "T")

BYTE_UNITS = {
    "B": 1,
    "KB": 1024,
    "MB": 1024 * 1024,
    "GB": 1024 * 1024 * 1024,
    "TB": 1024 * 1024 * 1024 * 1024,
}

T = TypeVar("T")


def human_byte_size(size: int, unit: str | None = None) -> str:
    r"""Gets a human-readable representation of the byte size.

    Args:
    ----
        size (int): Specifies the size in bytes.
        unit (str, optional): Specifies the unit. If ``None``, the
            best unit is found automatically. The supported units
            are: ``'B'``, ``'KB'``, ``'MB'``, ``'GB'``, ``'TB'``.
            Default: ``None``

    Returns:
    -------
        str: The byte size in a human-readable format.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.utils.format import human_byte_size
        >>> human_byte_size(2)
        '2.00 B'
        >>> human_byte_size(2048)
        '2.00 KB'
        >>> human_byte_size(2097152)
        '2.00 MB'
        >>> human_byte_size(2048, unit="B")
        '2,048.00 B'
    """
    if unit is None:  # Find the best unit.
        best_unit = "B"
        for unit, multiplier in BYTE_UNITS.items():
            if (size / multiplier) > 1:
                best_unit = unit
        unit = best_unit

    if unit not in BYTE_UNITS:
        raise ValueError(
            f"Incorrect unit '{unit}'. The available units are {list(BYTE_UNITS.keys())}"
        )

    return f"{size / BYTE_UNITS.get(unit, 1):,.2f} {unit}"


def human_count(number: int | float) -> str:
    r"""Converts an integer number with K, M, B, T for thousands,
    millions, billions and trillions, respectively.

    Args:
    ----
        number (int or float): A positive integer number. If the
            number is a float, it will be converted to an integer.

    Returns:
    -------
        str: A string formatted according to the pattern described
            above.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.utils.format import human_count
        >>> human_count(123)
        '123'
        >>> human_count(1234)  # (one thousand)
        '1.2 K'
        >>> human_count(2e6)  # (two million)
        '2.0 M'
        >>> human_count(3e9)  # (three billion)
        '3.0 B'
        >>> human_count(4e14)  # (four hundred trillion)
        '400 T'
        >>> human_count(5e15)  # (more than trillion)
        '5,000 T'
    """
    if number < 0:
        raise ValueError(f"The number should be a positive number (received {number})")
    if number < 1000:
        return str(int(number))
    labels = PARAMETER_NUM_UNITS
    num_digits = int(math.floor(math.log10(number)) + 1 if number > 0 else 1)
    num_groups = min(
        int(math.ceil(num_digits / 3)), len(labels)
    )  # don't abbreviate beyond trillions
    shift = -3 * (num_groups - 1)
    number = number * (10**shift)
    index = num_groups - 1
    if index < 1 or number >= 100:
        return f"{int(number):,d} {labels[index]}"
    return f"{number:,.1f} {labels[index]}"


def human_time(seconds: int | float) -> str:
    r"""Converts a number of seconds in an easier format to read
    hh:mm:ss.

    If the number of seconds is bigger than 1 day, this representation
    also encodes the number of days.

    Args:
    ----
        seconds (integer or float): Specifies the number of seconds.

    Returns:
    -------
        str: The number of seconds in a string format (hh:mm:ss).

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.utils.format import human_time
        >>> human_time(1.2)
        '0:00:01.200000'
        >>> human_time(61.2)
        '0:01:01.200000'
        >>> human_time(3661.2)
        '1:01:01.200000'
    """
    return str(datetime.timedelta(seconds=seconds))


def str_mapping(
    mapping: Mapping,
    sorted_keys: bool = False,
    num_spaces: int = 2,
    one_line: bool = False,
) -> str:
    r"""Computes a string representation of a mapping.

    Args:
    ----
        mapping (``Mapping``): Specifies the mapping.
        sorted_keys (bool, optional): Specifies if the key of the dict
            are sorted or not. Default: ``False``
        num_spaces (int, optional): Specifies the number of spaces
            used for the indentation. This option is used only when
            ``one_line=False``. Default: ``2``.
        one_line (bool, optional): If ``True``, tryes to generate a
            single line string representation. The keys and values
            should not contain multiple lines. If ``False``, a new
            line is created for each key in the input mapping.
            Default: ``False``

    Returns:
    -------
        str: The string representation of the mapping.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.utils.format import str_mapping
        >>> print(str_mapping({"key1": "value1", "key2": "value2"}))
        key1=value1
        key2=value2
        >>> print(str_mapping({"key1": "long\nvalue1", "key2": "value2"}))
        key1=long
          value1
        key2=value2
        >>> print(str_mapping({"key1": "value1", "key2": "value2"}, one_line=True))
        key1=value1, key2=value2
    """
    items = sorted(mapping.items()) if sorted_keys else mapping.items()
    if one_line:
        return ", ".join([f"{key}={value}" for key, value in items])
    return "\n".join([f"{key}={str_indent(value, num_spaces=num_spaces)}" for key, value in items])


def str_scalar(value: int | float) -> str:
    r"""Returns a string representation of a scalar value.

    Args:
    ----
        value (int or float): Specifies the input value.

    Returns:
    -------
        str: The string representation of the input value.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.utils.format import str_scalar
        >>> str_scalar(123456.789)
        123,456.789000
        >>> str_scalar(1234567)
        1,234,567
        >>> str_scalar(12345678901)
        1.234568e+10
    """
    if isinstance(value, int):
        if math.fabs(value) >= 1e9:
            return f"{value:.6e}"
        return f"{value:,}"
    if math.fabs(value) < 1e-3 or math.fabs(value) >= 1e6:
        return f"{value:.6e}"
    return f"{value:,.6f}"


def str_target_object(config: dict) -> str:
    r"""Gets a string that indicates the target object in the config.

    Args:
    ----
        config (dict): Specifies a config using the ``object_factory``
            library. This dict is expected to have a key
            ``'_target_'`` to indicate the target object.

    Returns:
    -------
        str: A string with the target object.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.utils.format import str_target_object
        >>> str_target_object({OBJECT_TARGET: "something.MyClass"})
        [_target_: something.MyClass]
        >>> str_target_object({})
        [_target_: N/A]
    """
    return f"[{OBJECT_TARGET}: {config.get(OBJECT_TARGET, 'N/A')}]"


def str_pretty_json(data: Any, sort_keys: bool = True, indent: int = 2, max_len: int = 80) -> str:
    r"""Converts a data structure to a pretty JSON string.

    Args:
    ----
        data: Specifies the input to convert to a pretty JSON string.
        sort_keys (bool, optional): Specifies if the keys are sorted
            or not. Default: ``True``
        indent (int, optional): Specifies the indent. It is a
            non-negative integer. Default: ``2``
        max_len (int, optional): Specifies the maximum length of the
            string representation. If the string representation is
            longer than this length, it is converted to the json
            representation. Default: ``80``

    Returns:
    -------
        str: The string representation.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.utils.format import str_pretty_json
        >>> str_pretty_json({"my_key": "my_value"})
        "{'my_key': 'my_value'}"
        >>> str_pretty_json(["value1", "value2"])
        "['value1', 'value2']"
        >>> str_pretty_json(["value1", "value2"], max_len=5)
        '[\n  "value1",\n  "value2"\n]'
    """
    str_data = str(data)
    if len(str_data) < max_len:
        return str_data
    return json.dumps(data, sort_keys=sort_keys, indent=indent, default=str)


def str_pretty_yaml(data: Any, sort_keys: bool = True, indent: int = 2, max_len: int = 80) -> str:
    r"""Converts a data structure to a pretty YAML string.

    Args:
    ----
        data: Specifies the input to convert to a pretty YAML string.
        sort_keys (bool, optional): Specifies if the keys are sorted
            or not. Default: ``True``
        indent (int, optional): Specifies the indent. It is a
            non-negative integer. Default: ``2``
        max_len (int, optional): Specifies the maximum length of the
            string representation. If the string representation is
            longer than this length, it is converted to the json
            representation. Default: ``max_len``

    Returns:
    -------
        str: The string representation.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.utils.format import str_pretty_yaml
        >>> str_pretty_yaml({"my_key": "my_value"})
        "{'my_key': 'my_value'}"
        >>> str_pretty_yaml(["value1", "value2"])
        "['value1', 'value2']"
        >>> str_pretty_yaml(["value1", "value2"], max_len=5)
        '- value1\n- value2\n'
    """
    str_data = str(data)
    if len(str_data) < max_len:
        return str_data
    return yaml.safe_dump(data, sort_keys=sort_keys, indent=indent)


def str_pretty_dict(data: dict[str, Any], sorted_keys: bool = False, indent: int = 0) -> str:
    r"""Converts a dict to a pretty string representation.

    This function was designed for flat dictionary. If you have a
    nested dictionary, you may consider other functions. Note that
    this function works for nested dict but the output may not be
    nice.

    Args:
    ----
        data (dict): Specifies the input dictionary.
        sorted_keys (bool, optional): Specifies if the key of the dict
            are sorted or not. Default: ``False``
        indent (int, optional): Specifies the indentation. The value
            should be greater or equal to 0. Default: ``0``

    Returns:
    -------
        str: The string representation.

        Example usage:

    .. code-block:: pycon

        >>> from gravitorch.utils.format import str_pretty_dict
        >>> str_pretty_dict({"my_key": "my_value"})
        'my_key : my_value'
        >>> str_pretty_dict({"key1": "value1", "key2": "value2"})
        'key1 : value1\nkey2 : value2'
    """
    if indent < 0:
        raise ValueError(f"The indent has to be greater or equal to 0 (received: {indent})")
    if not data:
        return ""

    max_length = max([len(key) for key in data])
    output = []
    for key in sorted(data.keys()) if sorted_keys else data.keys():
        output.append(f"{' ' * indent + str(key) + ' ' * (max_length - len(key))} : {data[key]}")
    return "\n".join(output)
