from __future__ import annotations

from unittest.mock import Mock, patch

from objectory import OBJECT_TARGET
from pytest import fixture, mark, raises
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.datapipes.iter import IterableWrapper
from torch.utils.data.datapipes.iter.combinatorics import ShufflerIterDataPipe
from torch.utils.data.graph import DataPipe

from gravitorch.data.dataloaders import (
    create_dataloader,
    create_dataloader2,
    is_dataloader2_config,
    is_dataloader_config,
)
from gravitorch.data.datasets import DummyMultiClassDataset
from gravitorch.testing import torchdata_available
from gravitorch.utils.imports import is_torchdata_available

if is_torchdata_available():
    from torchdata.dataloader2 import (
        DataLoader2,
        MultiProcessingReadingService,
        ReadingServiceInterface,
    )
    from torchdata.dataloader2.adapter import Adapter, Shuffle
else:  # pragma: no cover
    Adapter = "Adapter"
    MultiProcessingReadingService, Shuffle = Mock(), Mock()


#######################################
#     Tests for create_dataloader     #
#######################################


@fixture(scope="module")
def dataset() -> Dataset:
    return DummyMultiClassDataset(num_examples=10, num_classes=2, feature_size=4)


@mark.parametrize(
    "dataset",
    (
        DummyMultiClassDataset(num_examples=10, num_classes=2, feature_size=4),
        {
            OBJECT_TARGET: "gravitorch.data.datasets.DummyMultiClassDataset",
            "num_examples": 10,
            "num_classes": 2,
            "feature_size": 4,
        },
    ),
)
def test_create_dataloader_dataset(dataset: Dataset | dict) -> None:
    dataloader = create_dataloader(dataset)
    assert isinstance(dataloader, DataLoader)
    assert isinstance(dataloader.dataset, DummyMultiClassDataset)
    assert dataloader.batch_size == 1
    assert isinstance(dataloader.sampler, SequentialSampler)


@mark.parametrize("batch_size", (1, 2))
def test_create_dataloader_batch_size(dataset: Dataset, batch_size: int) -> None:
    dataloader = create_dataloader(dataset, batch_size=batch_size)
    assert isinstance(dataloader, DataLoader)
    assert isinstance(dataloader.dataset, DummyMultiClassDataset)
    assert dataloader.batch_size == batch_size


def test_create_dataloader_sampler(dataset: Dataset) -> None:
    dataloader = create_dataloader(dataset, sampler=RandomSampler(dataset))
    assert isinstance(dataloader, DataLoader)
    assert isinstance(dataloader.dataset, DummyMultiClassDataset)
    assert dataloader.batch_size == 1
    assert isinstance(dataloader.sampler, RandomSampler)


def test_create_dataloader_config(dataset: Dataset) -> None:
    dataloader = create_dataloader(
        dataset,
        sampler={OBJECT_TARGET: "torch.utils.data.RandomSampler", "data_source": dataset},
    )
    assert isinstance(dataloader, DataLoader)
    assert isinstance(dataloader.dataset, DummyMultiClassDataset)
    assert dataloader.batch_size == 1
    assert isinstance(dataloader.sampler, RandomSampler)


########################################
#     Tests for create_dataloader2     #
########################################


@torchdata_available
@mark.parametrize(
    "datapipe",
    (
        IterableWrapper(range(10)),
        {OBJECT_TARGET: "torch.utils.data.datapipes.iter.IterableWrapper", "iterable": range(10)},
    ),
)
def test_create_dataloader2_datapipe(datapipe: DataPipe | dict) -> None:
    dataloader = create_dataloader2(datapipe)
    assert isinstance(dataloader, DataLoader2)
    assert isinstance(dataloader.datapipe, IterableWrapper)
    assert dataloader.datapipe_adapter_fns is None
    assert dataloader.reading_service is None


@torchdata_available
@mark.parametrize(
    "datapipe_adapter_fn",
    (
        Shuffle(),
        {OBJECT_TARGET: "torchdata.dataloader2.adapter.Shuffle"},
    ),
)
def test_create_dataloader2_datapipe_adapter_fn(datapipe_adapter_fn: Adapter | dict) -> None:
    dataloader = create_dataloader2(
        IterableWrapper(range(10)), datapipe_adapter_fn=datapipe_adapter_fn
    )
    assert isinstance(dataloader, DataLoader2)
    assert isinstance(dataloader.datapipe, ShufflerIterDataPipe)
    assert len(dataloader.datapipe_adapter_fns) == 1
    assert isinstance(dataloader.datapipe_adapter_fns[0], Shuffle)
    assert dataloader.reading_service is None


@torchdata_available
@mark.parametrize(
    "reading_service",
    (
        MultiProcessingReadingService(),
        {OBJECT_TARGET: "torchdata.dataloader2.MultiProcessingReadingService"},
    ),
)
def test_create_dataloader2_reading_service(
    reading_service: ReadingServiceInterface | dict,
) -> None:
    dataloader = create_dataloader2(IterableWrapper(range(10)), reading_service=reading_service)
    assert isinstance(dataloader, DataLoader2)
    assert isinstance(dataloader.datapipe, IterableWrapper)
    assert dataloader.datapipe_adapter_fns is None
    assert isinstance(dataloader.reading_service, MultiProcessingReadingService)


def test_create_dataloader2_without_torchdata() -> None:
    with patch("gravitorch.utils.imports.is_torchdata_available", lambda *args: False):
        with raises(RuntimeError, match="`torchdata` package is required but not installed."):
            create_dataloader2(IterableWrapper(range(10)))


##########################################
#     Tests for is_dataloader_config     #
##########################################


def test_is_dataloader_config_true() -> None:
    assert is_dataloader_config({"_target_": "torch.utils.data.DataLoader"})


def test_is_dataloader_config_false() -> None:
    assert not is_dataloader_config({"_target_": "torch.nn.Identity"})


@torchdata_available
def test_is_dataloader_config_false_dataloader2() -> None:
    assert not is_dataloader_config({"_target_": "torchdata.dataloader2.DataLoader2"})


###########################################
#     Tests for is_dataloader2_config     #
###########################################


@torchdata_available
def test_is_dataloader2_config_true() -> None:
    assert is_dataloader2_config({"_target_": "torchdata.dataloader2.DataLoader2"})


@torchdata_available
def test_is_dataloader2_config_false() -> None:
    assert not is_dataloader2_config({"_target_": "torch.nn.Identity"})


@torchdata_available
def test_is_dataloader2_config_false_dataloader() -> None:
    assert not is_dataloader2_config({"_target_": "torch.utils.data.DataLoader"})
