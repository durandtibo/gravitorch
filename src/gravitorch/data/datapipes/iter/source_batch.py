__all__ = ["DictBatcherSrcIterDataPipe", "TupleBatcherSrcIterDataPipe"]

import logging
from collections.abc import Hashable, Iterator, Sequence

import torch
from torch import Tensor
from torch.utils.data import IterDataPipe

from gravitorch.data.datapipes.iter.shuffling import (
    shuffle_tensor_mapping,
    shuffle_tensors,
)
from gravitorch.utils.format import str_add_indent
from gravitorch.utils.seed import get_torch_generator
from gravitorch.utils.summary import concise_summary

logger = logging.getLogger(__name__)


class DictBatcherSrcIterDataPipe(IterDataPipe[dict]):
    r"""Implements a source DataPipe to generate batch of examples from a
    dictionary of ``torch.Tensor``s.

    Args:
        data (dict): Specifies a dictionary with the data. The
            generated batches have the same structure that this input.
        batch_size (int): Specifies the batch size.
        shuffle (bool, optional): If ``True``, the examples are
            shuffled before to create the batches. Default: ``False``
        random_seed (int, optional): Specifies the random seed used to
            shuffle the data. Default: ``11918852809641073385``
    """

    def __init__(
        self,
        data: dict[Hashable, Tensor],
        batch_size: int,
        shuffle: bool = False,
        random_seed: int = 11918852809641073385,
    ):
        self._data = data
        self._batch_size = int(batch_size)
        self._shuffle = bool(shuffle)
        self._generator = get_torch_generator(random_seed)

    def __iter__(self) -> Iterator[dict]:
        data = self._data
        if self._shuffle:
            data = shuffle_tensor_mapping(data, generator=self._generator)
        keys = data.keys()
        for batched_tensors in zip(
            *[torch.split(value, self._batch_size) for value in data.values()]
        ):
            yield {key: batched_tensor for key, batched_tensor in zip(keys, batched_tensors)}

    def __len__(self) -> int:
        return (
            self._data[next(iter(self._data))].shape[0] + self._batch_size - 1
        ) // self._batch_size

    def __str__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  data:\n    {str_add_indent(concise_summary(self._data), num_spaces=4)}\n"
            f"  batch_size={self._batch_size},\n"
            f"  shuffle={self._shuffle},\n"
            f"  random_seed={self.random_seed},\n)"
        )

    @property
    def random_seed(self) -> int:
        r"""int: The random seed used to initialize the pseudo random
        generator.
        """
        return self._generator.initial_seed()


class TupleBatcherSrcIterDataPipe(IterDataPipe[tuple[Tensor, ...]]):
    r"""Implements a source DataPipe to generate batch of examples from a tuple
    of ``torch.Tensor``s.

    Args:
        tensors (``torch.Tensor`` of shape ``(num_examples, *)`` where
            ``*`` means any number of dimensions): Specifies the
            tensors.
        batch_size (int): Specifies the batch size.
        shuffle (bool, optional): If ``True``, the examples are
            shuffled before to create the batches.
            Default: ``False``
        random_seed (int, optional): Specifies the random seed used to
            shuffle the data. Default: ``13382866045483866228``
    """

    def __init__(
        self,
        tensors: Sequence[Tensor],
        batch_size: int,
        shuffle: bool = False,
        random_seed: int = 13382866045483866228,
    ):
        self._tensors = tensors
        self._batch_size = int(batch_size)
        self._shuffle = bool(shuffle)
        self._generator = get_torch_generator(random_seed)

    def __iter__(self) -> Iterator[tuple[Tensor, ...]]:
        tensors = self._tensors
        if self._shuffle:
            tensors = shuffle_tensors(tensors, generator=self._generator)
        yield from zip(*[torch.split(tensor, self._batch_size) for tensor in tensors])

    def __len__(self) -> int:
        return (self._tensors[0].shape[0] + self._batch_size - 1) // self._batch_size

    def __str__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  data:\n    {str_add_indent(concise_summary(self._tensors), num_spaces=4)}\n"
            f"  batch_size={self._batch_size},\n"
            f"  shuffle={self._shuffle},\n"
            f"  random_seed={self.random_seed},\n)"
        )

    @property
    def random_seed(self) -> int:
        r"""int: The random seed used to initialize the
        pseudo random generator.
        """
        return self._generator.initial_seed()
