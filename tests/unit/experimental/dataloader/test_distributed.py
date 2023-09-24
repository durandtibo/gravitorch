from __future__ import annotations

from unittest.mock import Mock, patch

import torch
from coola import objects_are_equal
from objectory import OBJECT_TARGET
from pytest import fixture, mark
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.utils.data.dataloader import default_collate

from gravitorch.creators.dataset import DatasetCreator
from gravitorch.dataloaders.collators import PaddedSequenceCollator
from gravitorch.engines import BaseEngine
from gravitorch.experimental.dataloader import DistributedDataLoaderCreator


class FakeDataset(Dataset):
    def __len__(self) -> int:
        return 20

    def __getitem__(self, item: int) -> Tensor:
        return torch.ones(5).mul(item)


@fixture
def dataset() -> Dataset:
    return FakeDataset()


##################################################
#     Tests for DistributedDataLoaderCreator     #
##################################################


def test_distributed_dataloader_creator_str(dataset: Dataset) -> None:
    assert str(DistributedDataLoaderCreator(dataset)).startswith("DistributedDataLoaderCreator(")


def test_distributed_dataloader_creator_dataset(dataset: Dataset) -> None:
    dataloader = DistributedDataLoaderCreator(dataset, batch_size=8, shuffle=False).create()
    assert isinstance(dataloader, DataLoader)
    assert dataloader.dataset is dataset
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


def test_distributed_dataloader_creator_dataset_creator(dataset: Dataset) -> None:
    dataloader = DistributedDataLoaderCreator(
        DatasetCreator(dataset), batch_size=8, shuffle=False
    ).create()
    assert isinstance(dataloader, DataLoader)
    assert dataloader.dataset is dataset
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


@mark.parametrize("batch_size", (1, 2, 4))
def test_distributed_dataloader_creator_batch_size(dataset: Dataset, batch_size: int) -> None:
    dataloader = DistributedDataLoaderCreator(dataset, batch_size=batch_size).create()
    assert isinstance(dataloader, DataLoader)
    assert dataloader.batch_size == batch_size
    batch = next(iter(dataloader))
    assert torch.is_tensor(batch)
    assert batch.shape == (batch_size, 5)


def test_distributed_dataloader_creator_shuffle_false(dataset: Dataset) -> None:
    dataloader = DistributedDataLoaderCreator(dataset, batch_size=8, shuffle=False).create()
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
    dataloader = DistributedDataLoaderCreator(dataset, batch_size=8, shuffle=True).create()
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
    dataloader = DistributedDataLoaderCreator(dataset, num_workers=num_workers).create()
    assert isinstance(dataloader, DataLoader)
    assert dataloader.num_workers == num_workers


def test_distributed_dataloader_creator_pin_memory_false(dataset: Dataset) -> None:
    dataloader = DistributedDataLoaderCreator(dataset, pin_memory=False).create()
    assert isinstance(dataloader, DataLoader)
    assert not dataloader.pin_memory


def test_distributed_dataloader_creator_pin_memory_true(dataset: Dataset) -> None:
    dataloader = DistributedDataLoaderCreator(dataset, pin_memory=True).create()
    assert isinstance(dataloader, DataLoader)
    assert dataloader.pin_memory


def test_distributed_dataloader_creator_drop_last_false(dataset: Dataset) -> None:
    dataloader = DistributedDataLoaderCreator(dataset, drop_last=False).create()
    assert isinstance(dataloader, DataLoader)
    assert isinstance(dataloader.sampler, DistributedSampler)
    assert not dataloader.sampler.drop_last


def test_distributed_dataloader_creator_drop_last_true(dataset: Dataset) -> None:
    dataloader = DistributedDataLoaderCreator(dataset, drop_last=True).create()
    assert isinstance(dataloader, DataLoader)
    assert isinstance(dataloader.sampler, DistributedSampler)
    assert dataloader.sampler.drop_last


def test_distributed_dataloader_creator_reproduce(dataset: Dataset) -> None:
    creator = DistributedDataLoaderCreator(dataset, batch_size=8, shuffle=True)
    assert objects_are_equal(tuple(creator.create()), tuple(creator.create()))


def test_distributed_dataloader_creator_same_random_seed(dataset: Dataset) -> None:
    assert objects_are_equal(
        tuple(DistributedDataLoaderCreator(dataset, batch_size=8, shuffle=True, seed=1).create()),
        tuple(DistributedDataLoaderCreator(dataset, batch_size=8, shuffle=True, seed=1).create()),
    )


def test_distributed_dataloader_creator_different_random_seeds(dataset: Dataset) -> None:
    assert not objects_are_equal(
        tuple(DistributedDataLoaderCreator(dataset, batch_size=8, shuffle=True, seed=1).create()),
        tuple(DistributedDataLoaderCreator(dataset, batch_size=8, shuffle=True, seed=2).create()),
    )


def test_distributed_dataloader_creator_same_random_seed_same_epoch(
    dataset: Dataset,
) -> None:
    assert objects_are_equal(
        tuple(
            DistributedDataLoaderCreator(dataset, batch_size=8, shuffle=True, seed=1).create(
                engine=Mock(spec=BaseEngine, epoch=0)
            )
        ),
        tuple(
            DistributedDataLoaderCreator(dataset, batch_size=8, shuffle=True, seed=1).create(
                engine=Mock(spec=BaseEngine, epoch=0)
            )
        ),
    )


def test_distributed_dataloader_creator_same_random_seed_different_epochs(
    dataset: Dataset,
) -> None:
    assert not objects_are_equal(
        tuple(
            DistributedDataLoaderCreator(dataset, batch_size=8, shuffle=True, seed=1).create(
                engine=Mock(spec=BaseEngine, epoch=0)
            )
        ),
        tuple(
            DistributedDataLoaderCreator(dataset, batch_size=8, shuffle=True, seed=1).create(
                engine=Mock(spec=BaseEngine, epoch=1)
            )
        ),
    )


def test_distributed_dataloader_creator_collate_fn(dataset: Dataset) -> None:
    dataloader = DistributedDataLoaderCreator(dataset, collate_fn=default_collate).create()
    assert isinstance(dataloader, DataLoader)
    assert dataloader.collate_fn == default_collate


def test_distributed_dataloader_creator_collate_fn_none(dataset: Dataset) -> None:
    dataloader = DistributedDataLoaderCreator(dataset, collate_fn=None).create()
    assert isinstance(dataloader, DataLoader)
    assert dataloader.collate_fn == default_collate


def test_distributed_dataloader_creator_collate_fn_from_config(dataset: Dataset) -> None:
    dataloader = DistributedDataLoaderCreator(
        dataset,
        collate_fn={OBJECT_TARGET: "gravitorch.dataloaders.collators.PaddedSequenceCollator"},
    ).create()
    assert isinstance(dataloader, DataLoader)
    assert isinstance(dataloader.collate_fn, PaddedSequenceCollator)


@patch("gravitorch.experimental.dataloader.distributed.dist.get_world_size", lambda *args: 1)
def test_distributed_dataloader_creator_num_replicas_1(dataset: Dataset) -> None:
    dataloader = DistributedDataLoaderCreator(dataset, batch_size=1).create()
    assert isinstance(dataloader, DataLoader)
    assert len(dataloader) == 20


@patch("gravitorch.experimental.dataloader.distributed.dist.get_world_size", lambda *args: 2)
def test_distributed_dataloader_creator_num_replicas_2(dataset: Dataset) -> None:
    dataloader = DistributedDataLoaderCreator(dataset, batch_size=1).create()
    assert isinstance(dataloader, DataLoader)
    assert len(dataloader) == 10


@patch("gravitorch.experimental.dataloader.distributed.dist.get_world_size", lambda *args: 4)
def test_distributed_dataloader_creator_num_replicas_4(dataset: Dataset) -> None:
    dataloader = DistributedDataLoaderCreator(dataset, batch_size=1).create()
    assert isinstance(dataloader, DataLoader)
    assert len(dataloader) == 5


@patch("gravitorch.experimental.dataloader.distributed.dist.get_world_size", lambda *args: 2)
def test_distributed_dataloader_creator_num_replicas_2_ranks(dataset: Dataset) -> None:
    indices = set()
    with patch("gravitorch.experimental.dataloader.distributed.dist.get_rank", lambda *args: 0):
        for batch in DistributedDataLoaderCreator(dataset, batch_size=1).create():
            indices.add(batch[0, 0].item())
    with patch("gravitorch.experimental.dataloader.distributed.dist.get_rank", lambda *args: 1):
        for batch in DistributedDataLoaderCreator(dataset, batch_size=1).create():
            indices.add(batch[0, 0].item())
    assert len(indices) == 20


def test_distributed_dataloader_creator_dataset_repeat(dataset: Dataset) -> None:
    creator = DistributedDataLoaderCreator(dataset, batch_size=4, shuffle=True)
    dataloader1 = creator.create()
    assert isinstance(dataloader1, DataLoader)
    dataloader2 = creator.create()
    assert isinstance(dataloader2, DataLoader)

    assert dataloader1.dataset is dataloader2.dataset
    assert objects_are_equal(tuple(dataloader1), tuple(dataloader2))


def test_distributed_dataloader_creator_dataset_config_repeat() -> None:
    creator = DistributedDataLoaderCreator(
        {
            OBJECT_TARGET: "gravitorch.datasets.DummyMultiClassDataset",
            "num_examples": 10,
            "num_classes": 2,
            "feature_size": 4,
        },
        batch_size=4,
        shuffle=True,
    )
    dataloader1 = creator.create()
    assert isinstance(dataloader1, DataLoader)
    dataloader2 = creator.create()
    assert isinstance(dataloader2, DataLoader)

    assert dataloader1.dataset is not dataloader2.dataset
    assert objects_are_equal(tuple(dataloader1), tuple(dataloader2))


def test_distributed_dataloader_creator_dataset_creator_repeat_caching() -> None:
    creator = DistributedDataLoaderCreator(
        DatasetCreator(
            {
                OBJECT_TARGET: "gravitorch.datasets.DummyMultiClassDataset",
                "num_examples": 10,
                "num_classes": 2,
                "feature_size": 4,
            },
            cache=True,
        ),
        batch_size=4,
        shuffle=True,
    )
    dataloader1 = creator.create()
    assert isinstance(dataloader1, DataLoader)
    dataloader2 = creator.create()
    assert isinstance(dataloader2, DataLoader)

    assert dataloader1.dataset is dataloader2.dataset
    assert objects_are_equal(tuple(dataloader1), tuple(dataloader2))


def test_distributed_dataloader_creator_dataset_creator_repeat_no_caching() -> None:
    creator = DistributedDataLoaderCreator(
        DatasetCreator(
            {
                OBJECT_TARGET: "gravitorch.datasets.DummyMultiClassDataset",
                "num_examples": 10,
                "num_classes": 2,
                "feature_size": 4,
            },
        ),
        batch_size=4,
        shuffle=True,
    )
    dataloader1 = creator.create()
    assert isinstance(dataloader1, DataLoader)
    dataloader2 = creator.create()
    assert isinstance(dataloader2, DataLoader)

    assert dataloader1.dataset is not dataloader2.dataset
    assert objects_are_equal(tuple(dataloader1), tuple(dataloader2))
