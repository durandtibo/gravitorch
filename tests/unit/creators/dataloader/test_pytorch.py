from unittest.mock import Mock, patch

import torch
from coola import objects_are_equal
from objectory import OBJECT_TARGET
from pytest import fixture, mark
from torch import Tensor
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

    def __getitem__(self, item: int) -> Tensor:
        return torch.ones(5).mul(item)


@fixture
def dataset() -> Dataset:
    return FakeDataset()


###########################################
#     Tests for AutoDataLoaderCreator     #
###########################################


@mark.parametrize("batch_size", (None, 0, 1))
def test_auto_dataloader_creator_str(batch_size: int) -> None:
    assert str(AutoDataLoaderCreator(batch_size)).startswith("AutoDataLoaderCreator(")


@mark.parametrize("batch_size", (1, 2, 4))
@patch("gravitorch.creators.dataloader.pytorch.dist.is_distributed", lambda *args: False)
def test_auto_dataloader_creator_non_distributed(dataset: Dataset, batch_size: int) -> None:
    creator = AutoDataLoaderCreator(batch_size=batch_size)
    assert isinstance(creator._data_loader_creator, VanillaDataLoaderCreator)
    dataloader = creator.create(dataset)
    assert isinstance(dataloader, DataLoader)
    assert dataloader.batch_size == batch_size


@mark.parametrize("batch_size", (1, 2, 4))
@patch("gravitorch.creators.dataloader.pytorch.dist.is_distributed", lambda *args: True)
def test_auto_dataloader_creator_distributed(dataset: Dataset, batch_size: int) -> None:
    creator = AutoDataLoaderCreator(batch_size=batch_size)
    assert isinstance(creator._data_loader_creator, DistributedDataLoaderCreator)
    dataloader = creator.create(dataset)
    assert isinstance(dataloader, DataLoader)
    assert dataloader.batch_size == batch_size


##############################################
#     Tests for VanillaDataLoaderCreator     #
##############################################


def test_vanilla_dataloader_creator_str() -> None:
    assert str(VanillaDataLoaderCreator()).startswith("VanillaDataLoaderCreator(")


@mark.parametrize("batch_size", (1, 2, 4))
def test_vanilla_dataloader_creator_batch_size(dataset: Dataset, batch_size: int) -> None:
    dataloader = VanillaDataLoaderCreator(batch_size=batch_size).create(dataset)
    assert isinstance(dataloader, DataLoader)
    assert dataloader.batch_size == batch_size
    batch = next(iter(dataloader))
    assert torch.is_tensor(batch)
    assert batch.shape == (batch_size, 5)


def test_vanilla_dataloader_creator_shuffle_false(dataset: Dataset) -> None:
    dataloader = VanillaDataLoaderCreator(batch_size=8, shuffle=False).create(dataset)
    assert isinstance(dataloader, DataLoader)
    assert isinstance(dataloader.sampler, SequentialSampler)
    batch = next(iter(dataloader))
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


def test_vanilla_dataloader_creator_shuffle_true(dataset: Dataset) -> None:
    dataloader = VanillaDataLoaderCreator(batch_size=8, shuffle=True).create(dataset)
    assert isinstance(dataloader, DataLoader)
    assert isinstance(dataloader.sampler, RandomSampler)
    batch = next(iter(dataloader))
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
def test_vanilla_dataloader_creator_num_workers(dataset: Dataset, num_workers: int) -> None:
    dataloader = VanillaDataLoaderCreator(num_workers=num_workers).create(dataset)
    assert isinstance(dataloader, DataLoader)
    assert dataloader.num_workers == num_workers


def test_vanilla_dataloader_creator_pin_memory_false(dataset: Dataset) -> None:
    dataloader = VanillaDataLoaderCreator(pin_memory=False).create(dataset)
    assert isinstance(dataloader, DataLoader)
    assert not dataloader.pin_memory


def test_vanilla_dataloader_creator_pin_memory_true(dataset: Dataset) -> None:
    dataloader = VanillaDataLoaderCreator(pin_memory=True).create(dataset)
    assert isinstance(dataloader, DataLoader)
    assert dataloader.pin_memory


def test_vanilla_dataloader_creator_drop_last_false(dataset: Dataset) -> None:
    dataloader = VanillaDataLoaderCreator(drop_last=False).create(dataset)
    assert isinstance(dataloader, DataLoader)
    assert not dataloader.drop_last


def test_vanilla_dataloader_creator_drop_last_true(dataset: Dataset) -> None:
    dataloader = VanillaDataLoaderCreator(drop_last=True).create(dataset)
    assert isinstance(dataloader, DataLoader)
    assert dataloader.drop_last


def test_vanilla_dataloader_creator_same_random_seed(dataset: Dataset) -> None:
    assert objects_are_equal(
        tuple(DistributedDataLoaderCreator(batch_size=8, shuffle=True, seed=1).create(dataset)),
        tuple(DistributedDataLoaderCreator(batch_size=8, shuffle=True, seed=1).create(dataset)),
    )


def test_vanilla_dataloader_creator_different_random_seeds(dataset: Dataset) -> None:
    assert not objects_are_equal(
        tuple(DistributedDataLoaderCreator(batch_size=8, shuffle=True, seed=1).create(dataset)),
        tuple(DistributedDataLoaderCreator(batch_size=8, shuffle=True, seed=2).create(dataset)),
    )


def test_vanilla_dataloader_creator_same_random_seed_same_epoch(dataset: Dataset) -> None:
    assert objects_are_equal(
        tuple(
            DistributedDataLoaderCreator(batch_size=8, shuffle=True, seed=1).create(
                dataset, engine=Mock(spec=BaseEngine, epoch=0)
            )
        ),
        tuple(
            DistributedDataLoaderCreator(batch_size=8, shuffle=True, seed=1).create(
                dataset, engine=Mock(spec=BaseEngine, epoch=0)
            )
        ),
    )


def test_vanilla_dataloader_creator_same_random_seed_different_epochs(dataset: Dataset) -> None:
    assert not objects_are_equal(
        tuple(
            DistributedDataLoaderCreator(batch_size=8, shuffle=True, seed=1).create(
                dataset, engine=Mock(spec=BaseEngine, epoch=0)
            )
        ),
        tuple(
            DistributedDataLoaderCreator(batch_size=8, shuffle=True, seed=1).create(
                dataset, engine=Mock(spec=BaseEngine, epoch=1)
            )
        ),
    )


def test_vanilla_dataloader_creator_collate_fn(dataset: Dataset) -> None:
    dataloader = VanillaDataLoaderCreator(collate_fn=default_collate).create(dataset)
    assert isinstance(dataloader, DataLoader)
    assert dataloader.collate_fn == default_collate


def test_vanilla_dataloader_creator_collate_fn_none(dataset: Dataset) -> None:
    dataloader = VanillaDataLoaderCreator(collate_fn=None).create(dataset)
    assert isinstance(dataloader, DataLoader)
    assert dataloader.collate_fn == default_collate


def test_vanilla_dataloader_creator_collate_fn_from_config(dataset: Dataset) -> None:
    dataloader = VanillaDataLoaderCreator(
        collate_fn={OBJECT_TARGET: "gravitorch.data.dataloaders.collators.PaddedSequenceCollator"}
    ).create(dataset)
    assert isinstance(dataloader, DataLoader)
    assert isinstance(dataloader.collate_fn, PaddedSequenceCollator)


##################################################
#     Tests for DistributedDataLoaderCreator     #
##################################################


def test_distributed_dataloader_creator_str() -> None:
    assert str(DistributedDataLoaderCreator()).startswith("DistributedDataLoaderCreator(")


@mark.parametrize("batch_size", (1, 2, 4))
def test_distributed_dataloader_creator_batch_size(dataset: Dataset, batch_size: int) -> None:
    dataloader = DistributedDataLoaderCreator(batch_size=batch_size).create(dataset)
    assert isinstance(dataloader, DataLoader)
    assert dataloader.batch_size == batch_size
    batch = next(iter(dataloader))
    assert torch.is_tensor(batch)
    assert batch.shape == (batch_size, 5)


def test_distributed_dataloader_creator_shuffle_false(dataset: Dataset) -> None:
    dataloader = DistributedDataLoaderCreator(batch_size=8, shuffle=False).create(dataset)
    assert isinstance(dataloader, DataLoader)
    assert isinstance(dataloader.sampler, DistributedSampler)
    assert not dataloader.sampler.shuffle
    batch = next(iter(dataloader))
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


def test_distributed_dataloader_creator_shuffle_true(dataset: Dataset) -> None:
    dataloader = DistributedDataLoaderCreator(batch_size=8, shuffle=True).create(dataset)
    assert isinstance(dataloader, DataLoader)
    assert isinstance(dataloader.sampler, DistributedSampler)
    assert dataloader.sampler.shuffle
    batch = next(iter(dataloader))
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
def test_distributed_dataloader_creator_num_workers(dataset: Dataset, num_workers: int) -> None:
    dataloader = DistributedDataLoaderCreator(num_workers=num_workers).create(dataset)
    assert isinstance(dataloader, DataLoader)
    assert dataloader.num_workers == num_workers


def test_distributed_dataloader_creator_pin_memory_false(dataset: Dataset) -> None:
    dataloader = DistributedDataLoaderCreator(pin_memory=False).create(dataset)
    assert isinstance(dataloader, DataLoader)
    assert not dataloader.pin_memory


def test_distributed_dataloader_creator_pin_memory_true(dataset: Dataset) -> None:
    dataloader = DistributedDataLoaderCreator(pin_memory=True).create(dataset)
    assert isinstance(dataloader, DataLoader)
    assert dataloader.pin_memory


def test_distributed_dataloader_creator_drop_last_false(dataset: Dataset) -> None:
    dataloader = DistributedDataLoaderCreator(drop_last=False).create(dataset)
    assert isinstance(dataloader, DataLoader)
    assert not dataloader.sampler.drop_last


def test_distributed_dataloader_creator_drop_last_true(dataset: Dataset) -> None:
    dataloader = DistributedDataLoaderCreator(drop_last=True).create(dataset)
    assert isinstance(dataloader, DataLoader)
    assert dataloader.sampler.drop_last


def test_distributed_dataloader_creator_reproduce(dataset: Dataset) -> None:
    creator = DistributedDataLoaderCreator(batch_size=8, shuffle=True)
    assert objects_are_equal(tuple(creator.create(dataset)), tuple(creator.create(dataset)))


def test_distributed_dataloader_creator_same_random_seed(dataset: Dataset) -> None:
    assert objects_are_equal(
        tuple(DistributedDataLoaderCreator(batch_size=8, shuffle=True, seed=1).create(dataset)),
        tuple(DistributedDataLoaderCreator(batch_size=8, shuffle=True, seed=1).create(dataset)),
    )


def test_distributed_dataloader_creator_different_random_seeds(dataset: Dataset) -> None:
    assert not objects_are_equal(
        tuple(DistributedDataLoaderCreator(batch_size=8, shuffle=True, seed=1).create(dataset)),
        tuple(DistributedDataLoaderCreator(batch_size=8, shuffle=True, seed=2).create(dataset)),
    )


def test_distributed_dataloader_creator_same_random_seed_same_epoch(
    dataset: Dataset,
) -> None:
    assert objects_are_equal(
        tuple(
            DistributedDataLoaderCreator(batch_size=8, shuffle=True, seed=1).create(
                dataset, engine=Mock(spec=BaseEngine, epoch=0)
            )
        ),
        tuple(
            DistributedDataLoaderCreator(batch_size=8, shuffle=True, seed=1).create(
                dataset, engine=Mock(spec=BaseEngine, epoch=0)
            )
        ),
    )


def test_distributed_dataloader_creator_same_random_seed_different_epochs(
    dataset: Dataset,
) -> None:
    assert not objects_are_equal(
        tuple(
            DistributedDataLoaderCreator(batch_size=8, shuffle=True, seed=1).create(
                dataset, engine=Mock(spec=BaseEngine, epoch=0)
            )
        ),
        tuple(
            DistributedDataLoaderCreator(batch_size=8, shuffle=True, seed=1).create(
                dataset, engine=Mock(spec=BaseEngine, epoch=1)
            )
        ),
    )


def test_distributed_dataloader_creator_collate_fn(dataset: Dataset) -> None:
    dataloader = DistributedDataLoaderCreator(collate_fn=default_collate).create(dataset)
    assert isinstance(dataloader, DataLoader)
    assert dataloader.collate_fn == default_collate


def test_distributed_dataloader_creator_collate_fn_none(dataset: Dataset) -> None:
    dataloader = DistributedDataLoaderCreator(collate_fn=None).create(dataset)
    assert isinstance(dataloader, DataLoader)
    assert dataloader.collate_fn == default_collate


def test_distributed_dataloader_creator_collate_fn_from_config(dataset: Dataset) -> None:
    dataloader = DistributedDataLoaderCreator(
        collate_fn={OBJECT_TARGET: "gravitorch.data.dataloaders.collators.PaddedSequenceCollator"}
    ).create(dataset)
    assert isinstance(dataloader, DataLoader)
    assert isinstance(dataloader.collate_fn, PaddedSequenceCollator)


@patch("gravitorch.creators.dataloader.pytorch.dist.get_world_size", lambda *args: 1)
def test_distributed_dataloader_creator_num_replicas_1(dataset: Dataset) -> None:
    dataloader = DistributedDataLoaderCreator(batch_size=1).create(dataset)
    assert isinstance(dataloader, DataLoader)
    assert len(dataloader) == 20


@patch("gravitorch.creators.dataloader.pytorch.dist.get_world_size", lambda *args: 2)
def test_distributed_dataloader_creator_num_replicas_2(dataset: Dataset) -> None:
    dataloader = DistributedDataLoaderCreator(batch_size=1).create(dataset)
    assert isinstance(dataloader, DataLoader)
    assert len(dataloader) == 10


@patch("gravitorch.creators.dataloader.pytorch.dist.get_world_size", lambda *args: 4)
def test_distributed_dataloader_creator_num_replicas_4(dataset: Dataset) -> None:
    dataloader = DistributedDataLoaderCreator(batch_size=1).create(dataset)
    assert isinstance(dataloader, DataLoader)
    assert len(dataloader) == 5


@patch("gravitorch.creators.dataloader.pytorch.dist.get_world_size", lambda *args: 2)
def test_distributed_dataloader_creator_num_replicas_2_ranks(dataset: Dataset) -> None:
    indices = set()
    with patch("gravitorch.creators.dataloader.pytorch.dist.get_rank", lambda *args: 0):
        for batch in DistributedDataLoaderCreator(batch_size=1).create(dataset):
            indices.add(batch[0, 0].item())
    with patch("gravitorch.creators.dataloader.pytorch.dist.get_rank", lambda *args: 1):
        for batch in DistributedDataLoaderCreator(batch_size=1).create(dataset):
            indices.add(batch[0, 0].item())
    assert len(indices) == 20
