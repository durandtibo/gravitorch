__all__ = ["ClampTensorIterDataPipe", "ContiguousTensorIterDataPipe", "SymlogTensorIterDataPipe"]

from collections.abc import Iterator
from typing import TypeVar, Union

import torch
from torch.utils.data import IterDataPipe

from gravitorch.utils.format import str_add_indent
from gravitorch.utils.tensor import recursive_contiguous, symlog

T = TypeVar("T")


class ClampTensorIterDataPipe(IterDataPipe[dict]):
    r"""Implements an ``IterDataPipe`` to clamp/clip a ``torch.Tensor``.

    Args:
        source_datapipe (``IterDataPipe``): Specifies the source
            iterable DataPipe.
        key (str): Specifies the key associated to the tensor to
            clamp.
        min (float or int or ``None``): Specifies the minimum value.
            ``None`` means there is no minimum value.
            Default: ``None``
        max (float or int or ``None``): Specifies the maximum value.
            ``None`` means there is no maximum value.
            Default: ``None``
    """

    def __init__(
        self,
        source_datapipe: IterDataPipe[T],
        key: str,
        min: Union[float, int, None] = None,
        max: Union[float, int, None] = None,
    ):
        self._source_datapipe = source_datapipe
        self._key = key
        self._min = min
        self._max = max

        if min is None and max is None:
            raise ValueError("At least one of 'min' or 'max' must not be None")

    def __iter__(self) -> Iterator[T]:
        for data in self._source_datapipe:
            data[self._key].clamp_(min=self._min, max=self._max)
            yield data

    def __len__(self) -> int:
        try:
            return len(self._source_datapipe)
        except TypeError as exc:
            raise TypeError(
                f"{type(self).__qualname__} instance doesn't have valid length"
            ) from exc

    def __str__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  key={self._key},\n"
            f"  min={self._min},\n"
            f"  max={self._max},\n"
            f"  source_datapipe={str_add_indent(self._source_datapipe)},\n)"
        )


class ContiguousTensorIterDataPipe(IterDataPipe[T]):
    r"""Implements an ``IterDataPipe`` to return contiguous in memory tensors
    containing the same data as the input.

    Args:
        source_datapipe (``IterDataPipe``): Specifies the source
            iterable DataPipe.
        memory_format (``torch.memory_format``, optional): Specifies
            the desired memory format.
            Default: ``torch.contiguous_format``

    Example usage:

    .. code-block:: python

        >>> import torch
        >>> from gravitorch.data.datapipes.iter import ContiguousTensorIterDataPipe, SourceIterDataPipe
        >>> source = SourceIterDataPipe(
        ...     [torch.ones(3, 2).transpose(0, 1), torch.zeros(2, 4).transpose(0, 1)]
        ... )
        >>> [tensor.is_contiguous() for tensor in source]
        [False, False]
        >>> [tensor.is_contiguous() for tensor in ContiguousTensorIterDataPipe(source)]
        [True, True]
    """

    def __init__(
        self,
        source_datapipe: IterDataPipe[T],
        memory_format: torch.memory_format = torch.contiguous_format,
    ):
        self._source_datapipe = source_datapipe
        self._memory_format = memory_format

    def __iter__(self) -> Iterator[T]:
        for data in self._source_datapipe:
            yield recursive_contiguous(data, self._memory_format)

    def __len__(self) -> int:
        try:
            return len(self._source_datapipe)
        except TypeError as exc:
            raise TypeError(
                f"{type(self).__qualname__} instance doesn't have valid length"
            ) from exc

    def __str__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  memory_format={self._memory_format},\n"
            f"  source_datapipe={str_add_indent(self._source_datapipe)},\n)"
        )


class SymlogTensorIterDataPipe(ClampTensorIterDataPipe):
    r"""Implements an ``IterDataPipe`` to clamp/clip a ``torch.Tensor``, and
    then applies a symmetric log transformation."""

    def __iter__(self) -> Iterator[T]:
        for data in self._source_datapipe:
            data[self._key] = symlog(data[self._key].clamp(min=self._min, max=self._max))
            yield data
