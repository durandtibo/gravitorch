__all__ = ["concise_summary"]

from collections.abc import Mapping, Sequence
from itertools import islice
from typing import Any, Union

import numpy as np
import torch
from torch import Tensor

from gravitorch.utils.format import (
    str_indent,
    to_torch_mapping_str,
    to_torch_sequence_str,
)


def concise_summary(value: Any, num_spaces: int = 2, max_length: int = 5) -> str:
    r"""Creates a concise summary of a variable.

    This function supports the following types:

        - ``torch.Tensor``
        - ``numpy.ndarray``
        - ``list``
        - ``tuple``
        - ``dict``

    Args:
        value: Specifies the variable to summarize.
        num_spaces (int, optional): Specifies the number of spaces
            used for the indentation. Default: ``2``.
        max_length (int, optional): Specifies the number of values to
            show in the summary. This option is useful when the
            sequences/mappings are long because we may do not want to
            show all the values in the summary. Note this input is
            also used to create the summary of all the values in the
            sequence/mapping. Default: ``5``

    Returns:
        str: The concise summary of the variable.

    Example usage:

    .. code-block:: python

        >>> from gravitorch.utils.summary import concise_summary
        >>> print(concise_summary([1, 2, 3, 'abc', 5, 6, 7, 8, 9, 10]))
        <class 'list'> | length=10
          (0) <class 'int'> | value=1
          (1) <class 'int'> | value=2
          (2) <class 'int'> | value=3
          (3) <class 'str'> | value=abc
          (4) <class 'int'> | value=5
          ...
    """
    if torch.is_tensor(value) or isinstance(value, np.ndarray):
        return _concise_summary_tensor(value)
    if isinstance(value, (list, tuple)):
        return _concise_summary_sequence(
            sequence=value, num_spaces=num_spaces, max_length=max_length
        )
    if isinstance(value, dict):
        return _concise_summary_mapping(mapping=value, num_spaces=num_spaces, max_length=max_length)
    return f"{type(value)} | value={value}"


def _concise_summary_tensor(tensor: Union[Tensor, np.ndarray]) -> str:
    r"""Creates a concise summary of a ``torch.Tensor`` or ``numpy.ndarray``.

    Args:
       tensor: Specifies the tensor/array to summarize.

    Returns:
       str: The concise summary of the tensor/array.
    """
    summary = [f"{type(tensor)}", f"shape={tensor.shape}", f"dtype={tensor.dtype}"]
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    if tensor.dtype in (torch.half, torch.float, torch.double):
        summary.extend(
            [
                f"mean={tensor.mean():,.6f}",
                f"std={tensor.std():,.6f}",
                f"min={tensor.min():,.6f}",
                f"max={tensor.max():,.6f}",
            ]
        )
    if tensor.dtype in (torch.int8, torch.short, torch.int, torch.long):
        summary.extend(
            [
                f"mean={tensor.float().mean():,.6f}",
                f"std={tensor.float().std():,.6f}",
                f"min={tensor.min()}",
                f"max={tensor.max()}",
            ]
        )
    return " | ".join(summary)


def _concise_summary_sequence(sequence: Sequence, num_spaces: int = 2, max_length: int = 5) -> str:
    r"""Creates a concise summary of a sequence.

    Args:
        sequence: Specifies the sequence to summarize.
        num_spaces (int, optional): Specifies the number of spaces
            used for the indentation. Default: ``2``.
        max_length (int, optional): Specifies the number of values
            to show in the summary. This option is useful when the
            sequences are long because we may do not want to show
            all the values in the summary. Note this input is also
            used to create the summary of all the values in the
            sequence. Default: ``5``

    Returns:
        str: The concise summary of the sequence.
    """
    summary = to_torch_sequence_str(
        [
            concise_summary(val, num_spaces=num_spaces, max_length=max_length)
            for val in sequence[:max_length]
        ]
    )
    suffix = "\n..." if len(sequence) > max_length else ""
    return str_indent(
        f"{type(sequence)} | length={len(sequence):,}\n{summary}{suffix}", num_spaces=num_spaces
    )


def _concise_summary_mapping(mapping: Mapping, num_spaces: int = 2, max_length: int = 5) -> str:
    r"""Creates a concise summary of a mapping.

    Args:
        mapping: Specifies the mapping to summarize.
        num_spaces (int, optional): Specifies the number of spaces
            used for the indentation. Default: ``2``.
        max_length (int, optional): Specifies the number of values to
            show in the summary. This option is useful when the
            mapping are too big because we may do not want to show
            all the values in the summary. You can set this input to
            ``-1`` to show all the values in the summary. Note this
            input is also used to create the summary of all the values
            in the sequence. Default: ``5``

    Returns:
        str: The concise summary of the mapping.
    """
    summary = to_torch_mapping_str(
        {
            key: concise_summary(value, num_spaces=num_spaces, max_length=max_length)
            for key, value in islice(mapping.items(), max_length)
        }
    )
    suffix = "\n..." if len(mapping) > max_length else ""
    return str_indent(
        f"{type(mapping)} | length={len(mapping):,}\n{summary}{suffix}", num_spaces=num_spaces
    )
