__all__ = ["EpochShufflePartitioner"]

from collections.abc import Sequence
from typing import Optional, TypeVar, Union

import torch

from gravitorch.data.partitioners.base import BasePartitioner, setup_partitioner
from gravitorch.engines import BaseEngine
from gravitorch.utils.format import str_indent
from gravitorch.utils.seed import get_torch_generator

T = TypeVar("T")


class EpochShufflePartitioner(BasePartitioner[T]):
    r"""Implements a partitioner that shuffles the data before to partition
    them.

    To be reproducible, the shuffling is controlled by the engine
    epoch value and a base random seed. If no engine is provided,
    the epoch value is set to ``0``.

    Args:
        partitioner (``BasePartitioner`` or dict): Specifies a
            partitioner or its configuration.
        random_seed (int, optional): Specifies the base random seed.
            This random seed is added to the engine epoch value to
            define the initial seed of the ``torch.Generator`` object.
            Default: ``7553907118525846636``
    """

    def __init__(
        self, partitioner: Union[BasePartitioner, dict], random_seed: int = 7553907118525846636
    ):
        self._partitioner = setup_partitioner(partitioner)
        self._random_seed = random_seed

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  partitioner={str_indent(self._partitioner)},\n"
            f"  random_seed={self._random_seed},\n"
            ")"
        )

    @property
    def partitioner(self) -> BasePartitioner:
        r"""``BasePartitioner``: The partitioner."""
        return self._partitioner

    @property
    def random_seed(self) -> int:
        r"""int: The base random seed."""
        return self._random_seed

    def partition(
        self, items: Sequence[T], engine: Optional[BaseEngine] = None
    ) -> list[Sequence[T]]:
        epoch = engine.epoch if engine is not None else 0
        permutation = torch.randperm(
            len(items), generator=get_torch_generator(self._random_seed + epoch)
        )
        return self._partitioner.partition([items[i] for i in permutation], engine)
