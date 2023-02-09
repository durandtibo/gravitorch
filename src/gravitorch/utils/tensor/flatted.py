r"""This module implements some utility functions to use ``torch.Tensor``s."""

__all__ = ["LazyFlattedTensor"]

from typing import Any, Optional

import torch
from torch import Tensor

from gravitorch.distributed.ddp import all_gather_tensor_varshape
from gravitorch.utils.format import str_indent


class LazyFlattedTensor:
    r"""Implements a class to lazily concatenate flatted tensors.

    This class is at a very early stage and is very likely to change
    a lot in the future.

    To be more efficient, the tensors are concatenated only when the
    method ``consolidate`` is called. The tensors are stored in an
    internal buffer, then they are concatenated and stored in a
    separate variable. Storing the result in a separate variable
    leads to a more efficient design because the tensor is generated
    only one time. Adding another tensor leads to a new tensor when
    the method ``consolidate`` is called.

    Note: this class is independent of LazyTensor (see
    https://pytorch.org/blog/understanding-lazytensor-system-performance-with-pytorch-xla-on-cloud-tpu/
    for more information).

    Args:
        values (``torch.Tensor`` or ``None``, optional): Specifies the
            initial values. The tensor is flattened if necessary.
            ``None`` means no initial values. Default: ``None``

    Example usage:

    .. code-block:: python

        >>> import torch
        >>> from gravitorch.utils.tensor import LazyFlattedTensor
        >>> lazy_tensor = LazyFlattedTensor()
        >>> lazy_tensor.update(torch.arange(6))
        >>> lazy_tensor.update(torch.tensor([-3, 1, 7]))
        >>> lazy_tensor.values()
        tensor([ 0.,  1.,  2.,  3.,  4.,  5., -3.,  1.,  7.])
        >>> lazy_tensor.update(torch.arange(3))
        >>> lazy_tensor.values()
        tensor([ 0.,  1.,  2.,  3.,  4.,  5., -3.,  1.,  7.,  0.,  1.,  2.])
        # By default, the tensor type is torch.float32. To use another type like long,
        # you need to specify the target type when creating the LazyFlattedTensor object.
        >>> lazy_tensor = LazyFlattedTensor(torch.tensor([], dtype=torch.long))
        >>> lazy_tensor.update(torch.arange(6))
    """

    def __init__(self, values: Optional[Tensor] = None):
        if values is None:
            values = torch.tensor([])
        self._values = values.flatten()
        self._buffer = []

    def __str__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  values={str_indent(self._values)}\n"
            f"  buffer={str_indent(self._buffer)}\n)"
        )

    def all_reduce(self) -> "LazyFlattedTensor":
        r"""Reduces the values across all machines in such a way that all get
        the all the values.

        Returns:
            ``LazyFlattedTensor``: The reduced flatted tensor.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from gravitorch.utils.tensor import LazyFlattedTensor
            >>> lazy_tensor = LazyFlattedTensor()
            >>> lazy_tensor.update(torch.arange(6))
            >>> lazy_tensor_reduced = lazy_tensor.all_reduce()
        """
        return LazyFlattedTensor(torch.cat(all_gather_tensor_varshape(self.values()), dim=0))

    def clear(self) -> None:
        r"""Clears the values and the internal buffer."""
        self._values = torch.tensor([])
        self._buffer.clear()

    def clone(self) -> "LazyFlattedTensor":
        r"""Creates a copy of the current lazy flatted tensor.

        Returns:
            ``LazyFlattedTensor``: A copy of the current lazy flatted
                tensor.
        """
        return LazyFlattedTensor(self.values().clone())

    def consolidate(self) -> None:
        r"""Consolidates the current values and internal buffer in a single
        flatted tensor.

        This method does nothing if the lazy tensor is already
        consolidated.
        """
        if self._buffer:
            values = self._values
            if values.numel() == 0:
                # Use the first tensor in the buffer to find the initial data type.
                values = values.to(dtype=self._buffer[0].dtype)
            self._values = torch.cat(
                [values] + [tensor.flatten() for tensor in self._buffer],
                dim=0,
            )
            self._buffer.clear()

    def equal(self, other: Any) -> bool:
        r"""Indicates if two lazy flatted tensors are equal or not.

        Args:
            other: Specifies the value to compare.

        Returns:
            bool: ``True`` if the two lazy flatted tensors are equal,
                ``False`` otherwise.
        """
        if not isinstance(other, LazyFlattedTensor):
            return False
        if self.values().dtype != other.values().dtype:
            return False
        return self.values().equal(other.values())

    def numel(self) -> int:
        r"""Gets the total number of elements.

        Returns:
            int: The total number of elements.
        """
        return self._values.numel() + sum([tensor.numel() for tensor in self._buffer])

    def update(self, tensor: Tensor) -> None:
        r"""Updates the internal buffer by adding a new tensor.

        Args:
            tensor (``torch.Tensor``): Specifies the new tensor to add
                to the internal buffer. The tensor is flatted if
                necessary.
        """
        self._buffer.append(tensor)

    def values(self) -> Tensor:
        r"""Gets a flatted tensor with all the values.

        Returns:
            ``torch.Tensor``: The flatted tensor with all the values.
        """
        self.consolidate()
        return self._values
