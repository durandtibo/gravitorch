from __future__ import annotations

__all__ = [
    "AutoDataLoaderCreator",
    "VanillaDataLoaderCreator",
    "DistributedDataLoaderCreator",
]

from collections.abc import Callable
from typing import TypeVar

from torch.utils.data import DataLoader, Dataset, DistributedSampler

from gravitorch.creators.dataloader.base import BaseDataLoaderCreator
from gravitorch.data.dataloaders.factory import create_dataloader
from gravitorch.distributed import comm as dist
from gravitorch.engines.base import BaseEngine
from gravitorch.utils.format import str_indent, to_pretty_dict_str
from gravitorch.utils.seed import get_torch_generator

T = TypeVar("T")


class AutoDataLoaderCreator(BaseDataLoaderCreator[T]):
    r"""Defines a PyTorch data loader creator that automatically chooses
    the data loader creator based on the context.

    If the distributed package is activated, it uses the
    ``DistributedDataLoaderCreator``, otherwise it uses
    ``VanillaDataLoaderCreator``.


    Note the behavior of this class may change based on the new data
    loader creators.

    Args:
    ----
        batch_size (int, optional): Specifies the number of examples
            per batch to load. Default: ``1``
        shuffle (bool, optional): Specifies of the examples are
            shuffled or not. You should set to ``True`` to have the
            data reshuffled at every epoch. Default: ``False``
        num_workers (int, optional): Specifies the number of
            subprocesses to use for data loading. ``0`` means that
            the data will be loaded in the main process.
            Default: ``0``
        pin_memory (bool, optional): If ``True``, the data loader will
            copy Tensors into CUDA pinned memory before returning them.
            If your data elements are a custom type, or your
            :attr:`collate_fn` returns a batch that is a custom type,
            see the example below. Default: ``False``
        drop_last (bool, optional): set to ``True`` to drop the last
            incomplete batch, if the dataset size is not divisible by
            the batch size. If ``False`` and the size of dataset is
            not divisible by the batch size, then the last batch will
            be smaller. Default: ``False``
        seed (int, optional): Specifies the random seed used to
            shuffle the samples if ``shuffle=True``. Default: ``0``
        collate_fn (callable or dict or None, optional): Specifies the
            function used to merge a list of samples to form a
            mini-batch of Tensor(s). If ``None``, it uses the default
            PyTorch collate function. Default: ``None``
    """

    def __init__(
        self,
        batch_size: int | None = 1,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
        seed: int = 0,
        collate_fn: Callable | dict | None = None,
    ) -> None:
        if dist.is_distributed():
            self._data_loader_creator = DistributedDataLoaderCreator(
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=drop_last,
                seed=seed,
                collate_fn=collate_fn,
            )
        else:
            self._data_loader_creator = VanillaDataLoaderCreator(
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=drop_last,
                seed=seed,
                collate_fn=collate_fn,
            )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  data_loader_creator={str_indent(str(self._data_loader_creator))},\n"
            ")"
        )

    def create(self, dataset: Dataset, engine: BaseEngine | None = None) -> DataLoader[T]:
        return self._data_loader_creator.create(dataset=dataset, engine=engine)


class VanillaDataLoaderCreator(BaseDataLoaderCreator[T]):
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
            f"  {str_indent(to_pretty_dict_str(config, sorted_keys=True))}\n)"
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
            f"  {str_indent(to_pretty_dict_str(config, sorted_keys=True))}\n)"
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
        if engine is not None:
            # In distributed mode, calling the set_epoch() method at the beginning
            # of each epoch before creating the DataLoader iterator is necessary to
            # make shuffling work properly across multiple epochs.
            # Otherwise, the same ordering will always be used.
            sampler.set_epoch(engine.epoch)
        # Sampler option is mutually exclusive with shuffle or drop last.
        return create_dataloader(dataset, sampler=sampler, **self._kwargs)
