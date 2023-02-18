r"""This module defines some utility functions to format some objects."""

__all__ = [
    "human_byte_size",
    "human_count",
    "human_time",
    "str_indent",
    "str_scalar",
    "str_target_object",
    "to_flat_dict",
    "to_pretty_dict_str",
    "to_pretty_json_str",
    "to_pretty_yaml_str",
    "to_torch_mapping_str",
    "to_torch_sequence_str",
]

import datetime
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, Optional, TypeVar, Union

import yaml
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


def human_byte_size(size: int, unit: Optional[str] = None) -> str:
    r"""Gets a human-readable representation of the byte size.

    Args:
        size (int): Specifies the size in bytes.
        unit (str, optional): Specifies the unit. If ``None``, the
            best unit is found automatically. The supported units
            are: ``'B'``, ``'KB'``, ``'MB'``, ``'GB'``, ``'TB'``.
            Default: ``None``

    Returns:
        str: The byte size in a human-readable format.

    Example usage:

    .. code-block:: python

        >>> from gravitorch.utils.format import human_byte_size
        >>> human_byte_size(2)
        2.00 B
        >>> human_byte_size(2048)
        2.00 KB
        >>> human_byte_size(2097152)
        2.00 MB
        >>> human_byte_size(2048, unit='B')
        2,048.00 B
    """
    if unit is None:  # Find the best unit.
        best_unit = "B"
        for unit, multiplier in BYTE_UNITS.items():
            if (size / multiplier) > 1:
                best_unit = unit
        unit = best_unit

    if unit not in BYTE_UNITS:
        raise ValueError(
            f"Incorrect unit {unit}. The available units are {list(BYTE_UNITS.keys())}"
        )

    return f"{size / BYTE_UNITS.get(unit, 1):,.2f} {unit}"


def human_count(number: Union[int, float]) -> str:
    r"""Converts an integer number with K, M, B, T for thousands, millions,
    billions and trillions, respectively.

    Args:
        number (int or float): A positive integer number. If the
            number is a float, it will be converted to an integer.

    Returns:
        str: A string formatted according to the pattern described
            above.

    Example usage:

    .. code-block:: python

        >>> from gravitorch.utils.format import human_count
        >>> human_count(123)
        '123  '
        >>> human_count(1234)  # (one thousand)
        '1.2 K'
        >>> human_count(2e6)   # (two million)
        '2.0 M'
        >>> human_count(3e9)   # (three billion)
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


def human_time(seconds: Union[int, float]) -> str:
    r"""Converts a number of seconds in an easier format to read hh:mm:ss.

    If the number of seconds is bigger than 1 day, this representation
    also encodes the number of days.

    Args:
        seconds (integer or float): Specifies the number of seconds.

    Returns:
        str: The number of seconds in a string format (hh:mm:ss).

    Example usage:

    .. code-block:: python

        >>> from gravitorch.utils.format import human_time
        >>> human_time(1.2)
        '0:00:01.200000'
        >>> human_time(61.2)
        '0:01:01.200000'
        >>> human_time(3661.2)
        '1:01:01.200000'
    """
    return str(datetime.timedelta(seconds=seconds))


def str_indent(original: Any, num_spaces: int = 2) -> str:
    r"""Adds indentations if the original string is a multi-lines string.

    Args:
        original: Specifies the original string. If the inputis not a
            string, it will be converted to a string with the function
            ``str``.
        num_spaces (int, optional): Specifies the number of spaces
            used for the indentation. Default: ``2``.

    Returns:
        str: The indented string.

    Example usage:

    .. code-block:: python

        >>> string_to_print = "string1\nstring2"
        >>> print(f"{string_to_print}")
        string1
        string2
        >>> print(f"\t{string_to_print}")
            string1
        string2
        # The problem is that 'string1' and 'string2' are not aligned.
        # The indentation is only applied to the first line
        >>> from gravitorch.utils.format import str_indent
        >>> print(f"\t{str_indent(string_to_print, 4)}")
            string1
            string2
    """
    formatted_str = str(original).split("\n")
    if len(formatted_str) == 1:  # single line
        return formatted_str[0]
    first = formatted_str.pop(0)
    formatted_str = "\n".join([(num_spaces * " ") + line for line in formatted_str])
    return first + "\n" + formatted_str


def str_scalar(value: Union[int, float]) -> str:
    r"""Returns a string representation of a scalar value.

    Args:
        value (int or float): Specifies the input value.

    Returns:
        str: The string representation of the input value.

    Example usage:

    .. code-block:: python

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
        config (dict): Specifies a config using the ``object_factory``
            library. This dict is expected to have a key
            ``'_target_'`` to indicate the target object.

    Returns:
        str: A string with the target object.

    Example usage:

    .. code-block:: python

        >>> from gravitorch.utils.format import str_target_object
        >>> str_target_object({OBJECT_TARGET: "something.MyClass"})
        [_target_: something.MyClass]
        >>> str_target_object({})
        [_target_: N/A]
    """
    return f"[{OBJECT_TARGET}: {config.get(OBJECT_TARGET, 'N/A')}]"


def to_flat_dict(
    data: Union[dict, list, tuple, str, bool, int, float, None],
    prefix: Optional[str] = None,
    separator: str = ".",
    to_str: Union[type[object], tuple[type[object], ...], None] = None,
) -> dict[str, Union[str, bool, int, float, None]]:
    r"""Computes a flat representation of a nested dict with the dot format.

    Args:
        data: Specifies the nested dict to flat.
        prefix (str, optional): Specifies the prefix to use to
            generate the name of the key. ``None`` means no prefix.
            Default: ``None``
        separator (str, optional): Specifies the separator to
            concatenate keys of nested collections. Default: ``'.'``
        to_str (tuple or ``None``, optional): Specifies the data types
            which will not be flattened out, instead converted to
            string. Default: ``None``

    Returns:
        dict: The flatted data.

    Example usage:

    .. code-block:: python

        >>> from gravitorch.utils.format import to_flat_dict
        >>> data = {
                'str': 'def',
                'module': {
                    'component': {
                        'float': 3.5,
                        'int': 2,
                    },
                },
            }
        >>> to_flat_dict(data)
        {
            'module.component.float': 3.5,
            'module.component.int': 2,
            'str': 'def',
        }
        # Example with lists (also works with tuple)
        >>> data = {
                'module': [[1, 2, 3], {'bool': True}],
                'str': 'abc',
            }
        >>> to_flat_dict(data)
        {
            'module.0.0': 1,
            'module.0.1': 2,
            'module.0.2': 3,
            'module.1.bool': True,
            'str': 'abc',
        }

        # Example with lists with to_str=(list) (also works with tuple)
        >>> data = {
                'module': [[1, 2, 3], {'bool': True}],
                'str': 'abc',
            }
        >>> to_flat_dict(data)
        {
            'module': "[[1, 2, 3], {'bool': True}]",
            'str': 'abc',
        }
    """
    flat_dict = {}
    to_str = to_str or tuple()
    if isinstance(data, to_str):
        flat_dict[prefix] = str(data)
    elif isinstance(data, dict):
        for key, value in data.items():
            flat_dict.update(
                to_flat_dict(
                    value,
                    prefix=f"{prefix}{separator}{key}" if prefix else key,
                    separator=separator,
                    to_str=to_str,
                )
            )
    elif isinstance(data, (list, tuple)):
        for i, value in enumerate(data):
            flat_dict.update(
                to_flat_dict(
                    value,
                    prefix=f"{prefix}{separator}{i}" if prefix else str(i),
                    separator=separator,
                    to_str=to_str,
                )
            )
    else:
        flat_dict[prefix] = data
    return flat_dict


def to_pretty_json_str(
    data: Any, sort_keys: bool = True, indent: int = 2, max_len: int = 80
) -> str:
    r"""Converts a data structure to a pretty JSON string.

    Args:
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
        str: The string representation.

    Example usage:

    .. code-block:: python

        >>> from gravitorch.utils.format import to_pretty_json_str
        >>> to_pretty_json_str({"my_key": "my_value"})
        "{'my_key': 'my_value'}"
        >>> to_pretty_json_str(["value1", "value2"])
        "['value1', 'value2']"
        >>> to_pretty_json_str(["value1", "value2"], max_len=5)
        '[\n  "value1",\n  "value2"\n]'
    """
    str_data = str(data)
    if len(str_data) < max_len:
        return str_data
    return json.dumps(data, sort_keys=sort_keys, indent=indent, default=str)


def to_pretty_yaml_str(
    data: Any, sort_keys: bool = True, indent: int = 2, max_len: int = 80
) -> str:
    r"""Converts a data structure to a pretty YAML string.

    Args:
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
        str: The string representation.

    Example usage:

    .. code-block:: python

        >>> from gravitorch.utils.format import to_pretty_yaml_str
        >>> to_pretty_yaml_str({"my_key": "my_value"})
        "{'my_key': 'my_value'}"
        >>> to_pretty_yaml_str(["value1", "value2"])
        "['value1', 'value2']"
        >>> to_pretty_yaml_str(["value1", "value2"], max_len=5)
        '- value1\n- value2\n'
    """
    str_data = str(data)
    if len(str_data) < max_len:
        return str_data
    return yaml.safe_dump(data, sort_keys=sort_keys, indent=indent)


def to_pretty_dict_str(data: dict[str, Any], sorted_keys: bool = False, indent: int = 0) -> str:
    r"""Converts a dict to a pretty string representation.

    This function was designed for flat dictionary. If you have a
    nested dictionary, you may consider other functions. Note that
    this function works for nested dict but the output may not be
    nice.

    Args:
        data (dict): Specifies the input dictionary.
        sorted_keys (bool, optional): Specifies if the key of the dict
            are sorted or not. Default: ``False``
        indent (int, optional): Specifies the indentation. The value
            should be greater or equal to 0. Default: ``0``

    Returns:
        str: The string representation.

        Example usage:

    .. code-block:: python

        >>> from gravitorch.utils.format import to_pretty_dict_str
        >>> to_pretty_dict_str({"my_key": "my_value"})
        'my_key : my_value'
        >>> to_pretty_dict_str({"key1": "value1", "key2": "value2"})
        'key1 : value1\nkey2 : value2'
    """
    if indent < 0:
        raise ValueError(f"The indent has to be greater or equal to 0 (received: {indent})")
    if not data:
        return ""

    max_length = max([len(key) for key in data.keys()])
    output = []
    for key in sorted(data.keys()) if sorted_keys else data.keys():
        output.append(f"{' ' * indent + str(key) + ' ' * (max_length - len(key))} : {data[key]}")
    return "\n".join(output)


def to_torch_mapping_str(mapping: Mapping, sorted_keys: bool = False, num_spaces: int = 2) -> str:
    r"""Computes a torch-like (``torch.nn.Module``) string representation of a
    mapping.

    Args:
        mapping (``Mapping``): Specifies the mapping.
        sorted_keys (bool, optional): Specifies if the key of the dict
            are sorted or not. Default: ``False``
        num_spaces (int, optional): Specifies the number of spaces
            used for the indentation. Default: ``2``.

    Returns:
        str: The string representation of the mapping.

    Example usage:

    .. code-block:: python

        >>> from gravitorch.utils.format import to_torch_mapping_str
        >>> to_torch_mapping_str({'key1': 'abc', 'key2': 'something\nelse'})
        (key1) abc
        (key2) something
          else
    """
    lines = []
    for key, value in sorted(mapping.items()) if sorted_keys else mapping.items():
        lines.append(f"({key}) {str_indent(value, num_spaces=num_spaces)}")
    return "\n".join(lines)


def to_torch_sequence_str(sequence: Sequence, num_spaces: int = 2) -> str:
    r"""Computes a torch-like (``torch.nn.Module``) string representation of a
    sequence.

    Args:
        sequence (``Sequence``): Specifies the sequence.
        num_spaces (int, optional): Specifies the number of spaces
            used for the indentation. Default: ``2``.

    Returns:
        str: The string representation of the sequence.

    Example usage:

    .. code-block:: python

        >>> from gravitorch.utils.format import to_torch_sequence_str
        >>> to_torch_sequence_str(["abc", "something\nelse"])
        (0) abc
        (1) something
          else
    """
    lines = []
    for i, item in enumerate(sequence):
        lines.append(f"({i}) {str_indent(item, num_spaces=num_spaces)}")
    return "\n".join(lines)
