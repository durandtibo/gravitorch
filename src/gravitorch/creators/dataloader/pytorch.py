from __future__ import annotations

__all__ = [
    "AutoDataLoaderCreator",
    "DataLoaderCreator",
    "DistributedDataLoaderCreator",
]

from typing import TypeVar

from torch.utils.data import DataLoader, Dataset, DistributedSampler

from gravitorch.creators.dataloader.base import BaseDataLoaderCreator
from gravitorch.data.dataloaders.factory import create_dataloader
from gravitorch.distributed import comm as dist
from gravitorch.engines.base import BaseEngine
from gravitorch.utils.format import str_indent, str_pretty_dict
from gravitorch.utils.seed import get_torch_generator

T = TypeVar("T")


class AutoDataLoaderCreator(BaseDataLoaderCreator[T]):
    r"""Defines a PyTorch data loader creator that automatically chooses
    the data loader creator based on the context.

    If the distributed package is activated, it uses the
    ``DistributedDataLoaderCreator``, otherwise it uses
    ``DataLoaderCreator``.


    Note the behavior of this class may change based on the new data
    loader creators.

    Args:
    ----
        **kwargs: See ``DataLoaderCreator`` or
            ``DistributedDataLoaderCreator`` documentation.
    """

    def __init__(self, **kwargs) -> None:
        if dist.is_distributed():
            self._creator = DistributedDataLoaderCreator(**kwargs)
        else:
            self._creator = DataLoaderCreator(**kwargs)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(\n" f"  creator={str_indent(str(self._creator))},\n" ")"
        )

    def create(self, dataset: Dataset, engine: BaseEngine | None = None) -> DataLoader[T]:
        return self._creator.create(dataset=dataset, engine=engine)


class DataLoaderCreator(BaseDataLoaderCreator[T]):
    r"""Defines a simple PyTorch data loader creator.

    Note that this data loader creator uses the default samplers.
    If you need a different sampler, you will need to implement your
    own data loader creator.

    Args:
    ----
        seed (int, optional): Specifies the random seed used to
            reproduce the shuffling of the samples. Default: ``0``
        **kwargs: See ``torch.utils.data.DataLoader`` documentation.
    """

    def __init__(self, seed: int = 0, **kwargs) -> None:
        self._seed = int(seed)
        self._kwargs = kwargs

    def __repr__(self) -> str:
        config = {"seed": self._seed} | self._kwargs
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  {str_indent(str_pretty_dict(config, sorted_keys=True))}\n)"
        )

    def create(self, dataset: Dataset, engine: BaseEngine | None = None) -> DataLoader[T]:
        epoch = 0 if engine is None else engine.epoch
        return create_dataloader(
            dataset, generator=get_torch_generator(self._seed + epoch), **self._kwargs
        )


class DistributedDataLoaderCreator(BaseDataLoaderCreator[T]):
    r"""Defines a simple distributed PyTorch data loader creator.

    This data loader creator uses the ``gravitorch.distributed`` package
    to distribute the example per process. Note that this data loader
    creator uses the default samplers. If you need a different sampler,
    you will need to implement your own data loader creator.

    Args:
    ----
        shuffle (bool, optional): Specifies of the examples are
            shuffled or not. You should set to ``True`` to have the
            data reshuffled at every epoch. Default: ``False``
        drop_last (bool, optional): set to ``True`` to drop the last
            incomplete batch, if the dataset size is not divisible by
            the batch size. If ``False`` and the size of dataset is
            not divisible by the batch size, then the last batch will
            be smaller. Default: ``False``
        seed (int, optional): Specifies the random seed used to
            shuffle the samples if ``shuffle=True``. Default: ``0``
        **kwargs: See ``torch.utils.data.DataLoader`` documentation.
    """

    def __init__(
        self, shuffle: bool = True, drop_last: bool = False, seed: int = 0, **kwargs
    ) -> None:
        self._shuffle = bool(shuffle)
        self._drop_last = bool(drop_last)
        self._seed = int(seed)
        self._kwargs = kwargs

    def __repr__(self) -> str:
        config = {
            "shuffle": self._shuffle,
            "drop_last": self._drop_last,
            "seed": self._seed,
        } | self._kwargs
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  {str_indent(str_pretty_dict(config, sorted_keys=True))}\n)"
        )

    def create(self, dataset: Dataset, engine: BaseEngine | None = None) -> DataLoader[T]:
        sampler = DistributedSampler(
            dataset,
            shuffle=self._shuffle,
            drop_last=self._drop_last,
            seed=self._seed,
            rank=dist.get_rank(),
            num_replicas=dist.get_world_size(),
        )
        epoch = 0
        if engine is not None:
            epoch = engine.epoch
            # In distributed mode, calling the set_epoch() method at the beginning
            # of each epoch before creating the DataLoader iterator is necessary to
            # make shuffling work properly across multiple epochs.
            # Otherwise, the same ordering will always be used.
            sampler.set_epoch(epoch)

        # Sampler option is mutually exclusive with shuffle or drop last.
        return create_dataloader(
            dataset,
            sampler=sampler,
            generator=get_torch_generator(self._seed + epoch),
            **self._kwargs,
        )
