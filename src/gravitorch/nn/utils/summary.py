r"""Defines some utilities function to analyze a ``torch.nn.Module``.

Inspired from https://github.com/PyTorchLightning/pytorch-
lightning/blob/master/pytorch_lightning/core/memory.py
"""

__all__ = [
    "ModuleSummary",
    "multiline_format_dtype",
    "multiline_format_size",
    "parse_batch_dtype",
    "parse_batch_shape",
]

from typing import Any, Union

import torch
from torch.nn import Module
from torch.utils.hooks import RemovableHandle

from gravitorch.nn.utils.module_helpers import num_learnable_parameters, num_parameters

PARAMETER_NUM_UNITS = (" ", "K", "M", "B", "T")
UNKNOWN_SIZE = "?"
UNKNOWN_DTYPE = "?"


class ModuleSummary:
    r"""Summary class for a single layer in a ``torch.nn.Module``.

    It collects the following information:

    - Type of the layer (e.g. Linear, BatchNorm1d, ...)
    - Input shape
    - Output shape
    - Input data type
    - Output data type
    - Number of parameters
    - Number of learnable parameters

    The input and output shapes are only known after the example input
    array was passed through the model.

    Example::

        >>> model = torch.nn.Conv2d(3, 8, 3)
        >>> summary = ModuleSummary(model)
        >>> summary.num_parameters
        224
        >>> summary.num_learnable_parameters
        224
        >>> summary.layer_type
        'Conv2d'
        >>> output = model(torch.rand(1, 3, 5, 5))
        >>> summary.in_size
        [1, 3, 5, 5]
        >>> summary.out_size
        [1, 8, 3, 3]
        >>> summary.in_dtype
        torch.float32
        >>> summary.out_dtype
        torch.float32

    Args:
        module: A module to summarize
    """

    def __init__(self, module: Module):
        super().__init__()
        self._module = module
        self._hook_handle = self._register_hook()
        self._in_size = None
        self._out_size = None
        self._in_dtype = None
        self._out_dtype = None

    def __del__(self):
        self.detach_hook()

    def _register_hook(self) -> RemovableHandle:
        r"""Registers a hook on the module that computes the input and output
        size(s) on the first forward pass.

        If the hook is called, it will remove itself from the module,
        meaning that recursive models will only record their input and
        output shapes once.

        Return:
            ``RemovableHandle``: A handle for the installed hook.
        """

        def hook(module, inp, out):
            if len(inp) == 1:
                inp = inp[0]
            self._in_size = parse_batch_shape(inp)
            self._out_size = parse_batch_shape(out)
            self._in_dtype = parse_batch_dtype(inp)
            self._out_dtype = parse_batch_dtype(out)
            self._hook_handle.remove()

        return self._module.register_forward_hook(hook)

    def detach_hook(self) -> None:
        r"""Removes the forward hook if it was not already removed in the
        forward pass.

        Will be called after the summary is created.
        """
        self._hook_handle.remove()

    @property
    def in_dtype(self) -> Union[str, list]:
        return self._in_dtype or UNKNOWN_DTYPE

    @property
    def out_dtype(self) -> Union[str, list]:
        return self._out_dtype or UNKNOWN_DTYPE

    @property
    def in_size(self) -> Union[str, list]:
        return self._in_size or UNKNOWN_SIZE

    @property
    def out_size(self) -> Union[str, list]:
        return self._out_size or UNKNOWN_SIZE

    @property
    def layer_type(self) -> str:
        r"""str: The class name of the module."""
        return str(self._module.__class__.__qualname__)

    @property
    def num_parameters(self) -> int:
        r"""int: The number of parameters in this module."""
        return num_parameters(self._module)

    @property
    def num_learnable_parameters(self) -> int:
        r"""int: The number of learnable parameters in this module."""
        return num_learnable_parameters(self._module)


def parse_batch_shape(batch: Any) -> Union[str, tuple]:
    r"""Parses the shapes in the batch.

    For now, it only parses the shapes of a tensor, list of tensors
    and tuple of tensors.

    Args:
        batch: Specifies the batch to parse.

    Returns:
        str or tuple: The shapes in the batch or ``"?"`` if it cannot
            parse the input..
    """
    if hasattr(batch, "shape"):
        return tuple(batch.shape)
    if isinstance(batch, (list, tuple)):
        return tuple(parse_batch_shape(el) for el in batch)
    return UNKNOWN_SIZE


def parse_batch_dtype(batch: Any) -> Union[str, tuple]:
    r"""Parses the data types in the batch.

    For now, it only parses the data type of tensor, list of tensors
    and tuple of tensors.

    Args:
        batch: Specifies the batch to parse.

    Returns:
        str or tuple: The data types in the batch.
    """
    if torch.is_tensor(batch):
        return str(batch.dtype)
    if isinstance(batch, (list, tuple)):
        return tuple(parse_batch_dtype(el) for el in batch)
    return UNKNOWN_DTYPE


def multiline_format_size(rows: Union[str, tuple, list]) -> tuple[str, ...]:
    formatted_rows = []
    for row in rows:
        if isinstance(row, (list, tuple)) and isinstance(row[0], (list, tuple)):
            formatted_rows.append("\n".join([str(r) for r in row]))
        else:
            formatted_rows.append(str(row))
    return tuple(formatted_rows)


def multiline_format_dtype(rows: Union[str, tuple, list]) -> tuple[str, ...]:
    formatted_rows = []
    for row in rows:
        if isinstance(row, (list, tuple)):
            formatted_rows.append("\n".join([str(r) for r in row]))
        else:
            formatted_rows.append(str(row))
    return tuple(formatted_rows)
