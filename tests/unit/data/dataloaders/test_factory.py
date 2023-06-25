from __future__ import annotations

from unittest.mock import patch

from objectory import OBJECT_TARGET
from pytest import mark, raises
from torch.utils.data.datapipes.iter import IterableWrapper
from torch.utils.data.datapipes.iter.combinatorics import ShufflerIterDataPipe
from torch.utils.data.graph import DataPipe

from gravitorch.data.dataloaders import create_dataloader2
from gravitorch.testing import torchdata_available
from gravitorch.utils.imports import is_torchdata_available

if is_torchdata_available():
    from torchdata.dataloader2 import (
        DataLoader2,
        MultiProcessingReadingService,
        ReadingServiceInterface,
    )
    from torchdata.dataloader2.adapter import Adapter, Shuffle


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
