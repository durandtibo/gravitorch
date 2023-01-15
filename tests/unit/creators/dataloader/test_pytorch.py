from unittest.mock import Mock, patch

import torch
from objectory import OBJECT_TARGET
from pytest import fixture, mark
from torch.utils.data import (
    Dataset,
    DistributedSampler,
    RandomSampler,
    SequentialSampler,
)
from torch.utils.data.dataloader import DataLoader, default_collate

from gravitorch.creators.dataloader import (
    AutoDataLoaderCreator,
    DistributedDataLoaderCreator,
    VanillaDataLoaderCreator,
)
from gravitorch.data.dataloaders.collators import PaddedSequenceCollator
from gravitorch.engines import BaseEngine


class FakeDataset(Dataset):
    def __len__(self) -> int:
        return 20

    def __getitem__(self, item):
        return item * torch.ones(5)


@fixture
def dataset() -> Dataset:
    return FakeDataset()


###########################################
#     Tests for AutoDataLoaderCreator     #
###########################################


@mark.parametrize("batch_size", (None, 0, 1))
def test_auto_data_loader_creator_str(batch_size):
    assert str(AutoDataLoaderCreator(batch_size)).startswith("AutoDataLoaderCreator(")


@mark.parametrize("batch_size", (1, 2, 4))
@patch("gravitorch.creators.dataloader.pytorch.dist.is_distributed", lambda *args: False)
def test_auto_data_loader_creator_non_distributed(dataset, batch_size):
    creator = AutoDataLoaderCreator(batch_size=batch_size)
    assert isinstance(creator._data_loader_creator, VanillaDataLoaderCreator)
    data_loader = creator.create(dataset)
    assert isinstance(data_loader, DataLoader)
    assert data_loader.batch_size == batch_size


@mark.parametrize("batch_size", (1, 2, 4))
@patch("gravitorch.creators.dataloader.pytorch.dist.is_distributed", lambda *args: True)
def test_auto_data_loader_creator_distributed(dataset, batch_size):
    creator = AutoDataLoaderCreator(batch_size=batch_size)
    assert isinstance(creator._data_loader_creator, DistributedDataLoaderCreator)
    data_loader = creator.create(dataset)
    assert isinstance(data_loader, DataLoader)
    assert data_loader.batch_size == batch_size


##############################################
#     Tests for VanillaDataLoaderCreator     #
##############################################


def test_vanilla_data_loader_creator_str():
    assert str(VanillaDataLoaderCreator()).startswith("VanillaDataLoaderCreator(")


@mark.parametrize("batch_size", (1, 2, 4))
def test_vanilla_data_loader_creator_batch_size(dataset, batch_size):
    creator = VanillaDataLoaderCreator(batch_size=batch_size)
    assert creator._batch_size == batch_size
    data_loader = creator.create(dataset)
    assert data_loader.batch_size == batch_size
    batch = next(iter(data_loader))
    assert batch.shape == (batch_size, 5)


def test_vanilla_data_loader_creator_shuffle_false(dataset):
    creator = VanillaDataLoaderCreator(batch_size=8, shuffle=False)
    assert not creator._shuffle
    data_loader = creator.create(dataset)
    assert isinstance(data_loader.sampler, SequentialSampler)
    batch = next(iter(data_loader))
    assert batch.equal(
        torch.tensor(
            [
                [0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1],
                [2, 2, 2, 2, 2],
                [3, 3, 3, 3, 3],
                [4, 4, 4, 4, 4],
                [5, 5, 5, 5, 5],
                [6, 6, 6, 6, 6],
                [7, 7, 7, 7, 7],
            ],
            dtype=torch.float,
        )
    )


def test_vanilla_data_loader_creator_shuffle_true(dataset):
    creator = VanillaDataLoaderCreator(batch_size=8, shuffle=True)
    assert creator._shuffle
    data_loader = creator.create(dataset)
    assert isinstance(data_loader.sampler, RandomSampler)
    batch = next(iter(data_loader))
    assert not batch.equal(
        torch.tensor(
            [
                [0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1],
                [2, 2, 2, 2, 2],
                [3, 3, 3, 3, 3],
                [4, 4, 4, 4, 4],
                [5, 5, 5, 5, 5],
                [6, 6, 6, 6, 6],
                [7, 7, 7, 7, 7],
            ],
            dtype=torch.float,
        )
    )


@mark.parametrize("num_workers", (0, 1, 2))
def test_vanilla_data_loader_creator_num_workers(num_workers):
    creator = VanillaDataLoaderCreator(num_workers=num_workers)
    assert creator._num_workers == num_workers


def test_vanilla_data_loader_creator_pin_memory_false(dataset):
    creator = VanillaDataLoaderCreator(pin_memory=False)
    assert not creator._pin_memory
    data_loader = creator.create(dataset)
    assert not data_loader.pin_memory


def test_vanilla_data_loader_creator_pin_memory_true(dataset):
    creator = VanillaDataLoaderCreator(pin_memory=True)
    assert creator._pin_memory
    data_loader = creator.create(dataset)
    assert data_loader.pin_memory


def test_vanilla_data_loader_creator_drop_last_false(dataset):
    creator = VanillaDataLoaderCreator(drop_last=False)
    assert not creator._drop_last
    data_loader = creator.create(dataset)
    assert not data_loader.drop_last


def test_vanilla_data_loader_creator_drop_last_true(dataset):
    creator = VanillaDataLoaderCreator(drop_last=True)
    assert creator._drop_last
    data_loader = creator.create(dataset)
    assert data_loader.drop_last


def test_vanilla_data_loader_creator_same_random_seed(dataset):
    data_loader1 = VanillaDataLoaderCreator(batch_size=8, shuffle=True, seed=42).create(dataset)
    batch1 = next(iter(data_loader1))
    data_loader2 = VanillaDataLoaderCreator(batch_size=8, shuffle=True, seed=42).create(dataset)
    batch2 = next(iter(data_loader2))
    assert batch1.equal(batch2)


def test_vanilla_data_loader_creator_different_random_seeds(dataset):
    data_loader1 = VanillaDataLoaderCreator(batch_size=8, shuffle=True, seed=1).create(dataset)
    batch1 = next(iter(data_loader1))
    data_loader2 = VanillaDataLoaderCreator(batch_size=8, shuffle=True, seed=42).create(dataset)
    batch2 = next(iter(data_loader2))
    assert not batch1.equal(batch2)


def test_vanilla_data_loader_creator_same_random_seeds_different_epochs(dataset):
    data_loader1 = VanillaDataLoaderCreator(batch_size=8, shuffle=True, seed=42).create(
        dataset, engine=Mock(spec=BaseEngine, epoch=0)
    )
    batch1 = next(iter(data_loader1))
    data_loader2 = VanillaDataLoaderCreator(batch_size=8, shuffle=True, seed=42).create(
        dataset, engine=Mock(spec=BaseEngine, epoch=1)
    )
    batch2 = next(iter(data_loader2))
    assert not batch1.equal(batch2)


def test_vanilla_data_loader_creator_epoch_none(dataset):
    data_loader1 = VanillaDataLoaderCreator(batch_size=8, shuffle=True, seed=42).create(
        dataset, engine=None
    )
    batch1 = next(iter(data_loader1))
    data_loader2 = VanillaDataLoaderCreator(batch_size=8, shuffle=True, seed=42).create(
        dataset, engine=None
    )
    batch2 = next(iter(data_loader2))
    assert batch1.equal(batch2)


def test_vanilla_data_loader_creator_collate_fn(dataset):
    creator = VanillaDataLoaderCreator(collate_fn=default_collate)
    assert creator._collate_fn == default_collate
    data_loader = creator.create(dataset)
    assert data_loader.collate_fn == default_collate


def test_vanilla_data_loader_creator_collate_fn_none(dataset):
    creator = VanillaDataLoaderCreator(collate_fn=None)
    assert creator._collate_fn == default_collate
    data_loader = creator.create(dataset)
    assert data_loader.collate_fn == default_collate


def test_vanilla_data_loader_creator_collate_fn_from_config(dataset):
    creator = VanillaDataLoaderCreator(
        collate_fn={OBJECT_TARGET: "gravitorch.data.dataloaders.collators.PaddedSequenceCollator"}
    )
    assert isinstance(creator._collate_fn, PaddedSequenceCollator)
    data_loader = creator.create(dataset)
    assert isinstance(data_loader.collate_fn, PaddedSequenceCollator)


##################################################
#     Tests for DistributedDataLoaderCreator     #
##################################################


def test_distributed_data_loader_creator_str():
    assert str(DistributedDataLoaderCreator()).startswith("DistributedDataLoaderCreator(")


@mark.parametrize("batch_size", (1, 2, 4))
def test_distributed_data_loader_creator_batch_size(dataset, batch_size):
    creator = DistributedDataLoaderCreator(batch_size=batch_size)
    assert creator._batch_size == batch_size
    data_loader = creator.create(dataset)
    assert data_loader.batch_size == batch_size
    batch = next(iter(data_loader))
    assert batch.shape == (batch_size, 5)


def test_distributed_data_loader_creator_shuffle_false(dataset):
    creator = DistributedDataLoaderCreator(batch_size=8, shuffle=False)
    assert not creator._shuffle
    data_loader = creator.create(dataset)
    assert isinstance(data_loader.sampler, DistributedSampler)
    assert not data_loader.sampler.shuffle
    batch = next(iter(data_loader))
    assert batch.equal(
        torch.tensor(
            [
                [0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1],
                [2, 2, 2, 2, 2],
                [3, 3, 3, 3, 3],
                [4, 4, 4, 4, 4],
                [5, 5, 5, 5, 5],
                [6, 6, 6, 6, 6],
                [7, 7, 7, 7, 7],
            ],
            dtype=torch.float,
        )
    )


def test_distributed_data_loader_creator_shuffle_true(dataset):
    creator = DistributedDataLoaderCreator(batch_size=8, shuffle=True)
    assert creator._shuffle
    data_loader = creator.create(dataset)
    assert isinstance(data_loader.sampler, DistributedSampler)
    assert data_loader.sampler.shuffle
    batch = next(iter(data_loader))
    assert not batch.equal(
        torch.tensor(
            [
                [0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1],
                [2, 2, 2, 2, 2],
                [3, 3, 3, 3, 3],
                [4, 4, 4, 4, 4],
                [5, 5, 5, 5, 5],
                [6, 6, 6, 6, 6],
                [7, 7, 7, 7, 7],
            ],
            dtype=torch.float,
        )
    )


@mark.parametrize("num_workers", (0, 1, 2))
def test_distributed_data_loader_creator_num_workers(num_workers):
    creator = DistributedDataLoaderCreator(num_workers=num_workers)
    assert creator._num_workers == num_workers


def test_distributed_data_loader_creator_pin_memory_false(dataset):
    creator = DistributedDataLoaderCreator(pin_memory=False)
    assert not creator._pin_memory
    data_loader = creator.create(dataset)
    assert not data_loader.pin_memory


def test_distributed_data_loader_creator_pin_memory_true(dataset):
    creator = DistributedDataLoaderCreator(pin_memory=True)
    assert creator._pin_memory
    data_loader = creator.create(dataset)
    assert data_loader.pin_memory


def test_distributed_data_loader_creator_drop_last_false(dataset):
    creator = DistributedDataLoaderCreator(drop_last=False)
    assert not creator._drop_last
    data_loader = creator.create(dataset)
    assert not data_loader.sampler.drop_last


def test_distributed_data_loader_creator_drop_last_true(dataset):
    creator = DistributedDataLoaderCreator(drop_last=True)
    assert creator._drop_last
    data_loader = creator.create(dataset)
    assert data_loader.sampler.drop_last


def test_distributed_data_loader_creator_same_random_seed(dataset):
    data_loader1 = DistributedDataLoaderCreator(batch_size=8, shuffle=True, seed=42).create(dataset)
    batch1 = next(iter(data_loader1))
    data_loader2 = DistributedDataLoaderCreator(batch_size=8, shuffle=True, seed=42).create(dataset)
    batch2 = next(iter(data_loader2))
    assert batch1.equal(batch2)


def test_distributed_data_loader_creator_different_random_seeds(dataset):
    data_loader1 = DistributedDataLoaderCreator(batch_size=8, shuffle=True, seed=1).create(dataset)
    batch1 = next(iter(data_loader1))
    data_loader2 = DistributedDataLoaderCreator(batch_size=8, shuffle=True, seed=42).create(dataset)
    batch2 = next(iter(data_loader2))
    assert not batch1.equal(batch2)


def test_distributed_data_loader_creator_same_random_seed_different_epochs(dataset):
    data_loader1 = DistributedDataLoaderCreator(batch_size=8, shuffle=True, seed=42).create(
        dataset, engine=Mock(spec=BaseEngine, epoch=0)
    )
    batch1 = next(iter(data_loader1))
    data_loader2 = DistributedDataLoaderCreator(batch_size=8, shuffle=True, seed=42).create(
        dataset, engine=Mock(spec=BaseEngine, epoch=1)
    )
    batch2 = next(iter(data_loader2))
    assert not batch1.equal(batch2)


def test_distributed_data_loader_creator_same_random_seed_engine_none(dataset):
    data_loader1 = DistributedDataLoaderCreator(batch_size=8, shuffle=True, seed=42).create(
        dataset, engine=None
    )
    batch1 = next(iter(data_loader1))
    data_loader2 = DistributedDataLoaderCreator(batch_size=8, shuffle=True, seed=42).create(
        dataset, engine=None
    )
    batch2 = next(iter(data_loader2))
    assert batch1.equal(batch2)


def test_distributed_data_loader_creator_collate_fn(dataset):
    creator = DistributedDataLoaderCreator(collate_fn=default_collate)
    assert creator._collate_fn == default_collate
    data_loader = creator.create(dataset)
    assert data_loader.collate_fn == default_collate


def test_distributed_data_loader_creator_collate_fn_none(dataset):
    creator = DistributedDataLoaderCreator(collate_fn=None)
    assert creator._collate_fn == default_collate
    data_loader = creator.create(dataset)
    assert data_loader.collate_fn == default_collate


def test_distributed_data_loader_creator_collate_fn_from_config(dataset):
    creator = DistributedDataLoaderCreator(
        collate_fn={OBJECT_TARGET: "gravitorch.data.dataloaders.collators.PaddedSequenceCollator"}
    )
    assert isinstance(creator._collate_fn, PaddedSequenceCollator)
    data_loader = creator.create(dataset)
    assert isinstance(data_loader.collate_fn, PaddedSequenceCollator)


@patch("gravitorch.creators.dataloader.pytorch.dist.get_world_size", lambda *args: 1)
def test_distributed_data_loader_creator_num_replicas_1(dataset):
    creator = DistributedDataLoaderCreator(batch_size=1)
    data_loader = creator.create(dataset)
    assert len(data_loader) == 20


@patch("gravitorch.creators.dataloader.pytorch.dist.get_world_size", lambda *args: 2)
def test_distributed_data_loader_creator_num_replicas_2(dataset):
    creator = DistributedDataLoaderCreator(batch_size=1)
    data_loader = creator.create(dataset)
    assert len(data_loader) == 10


@patch("gravitorch.creators.dataloader.pytorch.dist.get_world_size", lambda *args: 4)
def test_distributed_data_loader_creator_num_replicas_4(dataset):
    creator = DistributedDataLoaderCreator(batch_size=1)
    data_loader = creator.create(dataset)
    assert len(data_loader) == 5


@patch("gravitorch.creators.dataloader.pytorch.dist.get_world_size", lambda *args: 2)
def test_distributed_data_loader_creator_num_replicas_2_ranks(dataset):
    indices = set()
    with patch("gravitorch.creators.dataloader.pytorch.dist.get_rank", lambda *args: 0):
        for batch in DistributedDataLoaderCreator(batch_size=1).create(dataset):
            indices.add(batch[0, 0].item())
    with patch("gravitorch.creators.dataloader.pytorch.dist.get_rank", lambda *args: 1):
        for batch in DistributedDataLoaderCreator(batch_size=1).create(dataset):
            indices.add(batch[0, 0].item())
    assert len(indices) == 20
