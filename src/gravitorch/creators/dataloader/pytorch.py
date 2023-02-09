__all__ = [
    "AutoDataLoaderCreator",
    "VanillaDataLoaderCreator",
    "DistributedDataLoaderCreator",
]

from collections.abc import Callable
from typing import Optional, TypeVar, Union

import torch
from torch.utils.data import DataLoader, Dataset

from gravitorch.creators.dataloader.base import BaseDataLoaderCreator
from gravitorch.data.dataloaders.collators.utils import setup_collator
from gravitorch.distributed import comm as dist
from gravitorch.engines.base import BaseEngine
from gravitorch.utils.format import str_indent

T = TypeVar("T")


class AutoDataLoaderCreator(BaseDataLoaderCreator[T]):
    r"""Defines a PyTorch data loader creator that automatically chooses the
    data loader creator based on the context.

    If the distributed package is activated, it uses the
    ``DistributedDataLoaderCreator``, otherwise it uses
    ``VanillaDataLoaderCreator``.


    Note the behavior of this class may change based on the new data
    loader creators.

    Args:
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
        batch_size: Optional[int] = 1,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
        seed: int = 0,
        collate_fn: Union[Callable, dict, None] = None,
    ):
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

    def create(self, dataset: Dataset, engine: Optional[BaseEngine] = None) -> DataLoader[T]:
        return self._data_loader_creator.create(dataset=dataset, engine=engine)


class VanillaDataLoaderCreator(BaseDataLoaderCreator[T]):
    r"""Defines a simple PyTorch data loader creator.

    Note that this data loader creator uses the default samplers.
    If you need a different sampler, you will need to implement your
    own data loader creator.

    Args:
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
        batch_size: Optional[int] = 1,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
        seed: int = 0,
        collate_fn: Union[Callable, dict, None] = None,
    ):
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._num_workers = num_workers
        self._pin_memory = pin_memory
        self._drop_last = drop_last
        self._seed = seed

        self._collate_fn = setup_collator(collate_fn)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  batch_size={self._batch_size},\n"
            f"  shuffle={self._shuffle},\n"
            f"  num_workers={self._num_workers:,},\n"
            f"  pin_memory={self._pin_memory},\n"
            f"  drop_last={self._drop_last},\n"
            f"  seed={self._seed},\n"
            f"  collate_fn={str_indent(str(self._collate_fn))},\n"
            ")"
        )

    def create(self, dataset: Dataset, engine: Optional[BaseEngine] = None) -> DataLoader[T]:
        generator = torch.Generator()
        epoch = 0 if engine is None else engine.epoch
        generator.manual_seed(self._seed + epoch)
        return DataLoader(
            dataset=dataset,
            batch_size=self._batch_size,
            shuffle=self._shuffle,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            drop_last=self._drop_last,
            collate_fn=self._collate_fn,
            generator=generator,
        )


class DistributedDataLoaderCreator(VanillaDataLoaderCreator[T]):
    r"""Defines a simple distributed PyTorch data loader creator.

    This data loader creator uses the ``gravitorch.distributed`` package
    to distribute the example per process. Note that this data loader
    creator uses the default samplers. If you need a different sampler,
    you will need to implement your own data loader creator.
    """

    def create(self, dataset: Dataset, engine: Optional[BaseEngine] = None) -> DataLoader[T]:
        sampler = torch.utils.data.distributed.DistributedSampler(
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
        # Sampler option is mutually exclusive with shuffle.
        return DataLoader(
            dataset=dataset,
            sampler=sampler,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            collate_fn=self._collate_fn,
        )
