__all__ = ["PartialTransposerIterDataPipe", "TransposerIterDataPipe"]

from collections.abc import Iterator
from typing import TypeVar

from torch.utils.data import IterDataPipe

from gravitorch.utils.format import str_add_indent
from gravitorch.utils.tensor import partial_transpose_dict, recursive_transpose

T = TypeVar("T")


class TransposerIterDataPipe(IterDataPipe[T]):
    r"""Implements a source DataPipe to transpose all the ``torch.Tensor``s.

    Note: all the tensors should be compatible with the transpose
    dimensions.

    Args:
        source_datapipe (``IterDataPipe``): Specifies the source
            iterable DataPipe.
        dim0 (int): Specifies the first dimension to be transposed.
        dim1 (int): Specifies the second dimension to be transposed.

    Example usage:

    .. code-block:: python

        >>> import torch
        >>> from gravitorch.data.datapipes.iter import SourceIterDataPipe, TransposerIterDataPipe
        >>> source = SourceIterDataPipe([torch.ones(3, 2), torch.zeros(2, 4, 1)])
        >>> list(TransposerIterDataPipe(source, dim0=0, dim1=1))
        [
            tensor([[1., 1., 1.],
                    [1., 1., 1.]]),
            tensor([[[0.],
                     [0.]],
                    [[0.],
                     [0.]],
                    [[0.],
                     [0.]],
                    [[0.],
                     [0.]]]),
        ]
    """

    def __init__(self, source_datapipe: IterDataPipe[T], dim0: int, dim1: int):
        self._source_datapipe = source_datapipe
        self._dim0 = int(dim0)
        self._dim1 = int(dim1)

    def __iter__(self) -> Iterator[T]:
        for data in self._source_datapipe:
            yield recursive_transpose(data, dim0=self._dim0, dim1=self._dim1)

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
            f"  dim0={self._dim0},\n"
            f"  dim1={self._dim1},\n"
            f"  source_datapipe={str_add_indent(self._source_datapipe)},\n)"
        )

    @property
    def dim0(self) -> int:
        r"""int: The first dimension to be transposed."""
        return self._dim0

    @property
    def dim1(self) -> int:
        r"""int: The second dimension to be transposed."""
        return self._dim1


class PartialTransposerIterDataPipe(IterDataPipe[dict]):
    r"""Implements a source DataPipe to transpose all the ``torch.Tensor``s.

    Unlike ``TransposerIterDataPipe``, this DataPipe allows
    transposing only some tensors. It is possible to use different
    transpose dimensions.

    Args:
        source_datapipe (``IterDataPipe``): Specifies the source
            iterable DataPipe.
        dims (dict): Specifies the tensors to transpose, and the
            transposition dimensions. The keys should exist in each
            item of the source DataPipe. The keys indicate the tensors
            to transpose, and the values indicate the dimension to
            transpose. See ``partial_transpose_dict`` documentation
            for more details.

    Example usage:

    .. code-block:: python

        >>> import torch
        >>> from gravitorch.data.datapipes.iter import SourceIterDataPipe, TransposerIterDataPipe
        >>> source = SourceIterDataPipe([torch.ones(3, 2), torch.zeros(2, 4, 1)])
        >>> list(TransposerIterDataPipe(source, dim0=0, dim1=1))
        [
            tensor([[1., 1., 1.],
                    [1., 1., 1.]]),
            tensor([[[0.],
                     [0.]],
                    [[0.],
                     [0.]],
                    [[0.],
                     [0.]],
                    [[0.],
                     [0.]]]),
        ]
    """

    def __init__(self, source_datapipe: IterDataPipe[dict], dims: dict):
        self._source_datapipe = source_datapipe
        self._dims = dims

    def __iter__(self) -> Iterator[dict]:
        for data in self._source_datapipe:
            yield partial_transpose_dict(data, self._dims)

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
            f"  dims={self._dims},\n"
            f"  source_datapipe={str_add_indent(self._source_datapipe)},\n)"
        )
