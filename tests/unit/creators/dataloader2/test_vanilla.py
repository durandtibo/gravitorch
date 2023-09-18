from __future__ import annotations

from unittest.mock import Mock

from coola import objects_are_equal
from objectory import OBJECT_TARGET
from pytest import fixture, mark
from torch.utils.data import IterDataPipe, MapDataPipe
from torch.utils.data.datapipes.iter import IterableWrapper, Shuffler
from torch.utils.data.datapipes.map import SequenceWrapper

from gravitorch.creators.dataloader2 import (
    DataLoader2Creator,
    VanillaDataLoader2Creator,
)
from gravitorch.creators.datapipe import ChainedDataPipeCreator
from gravitorch.engines import BaseEngine
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


@fixture
def datapipe() -> IterDataPipe:
    return IterableWrapper([1, 2, 3, 4, 5])


DATAPIPES = (IterableWrapper([1, 2, 3, 4, 5]), SequenceWrapper([1, 2, 3, 4, 5]))


########################################
#     Tests for DataLoader2Creator     #
########################################


@torchdata_available
def test_dataloader_creator_str() -> None:
    assert str(DataLoader2Creator(Mock(spec=DataLoader2))).startswith("DataLoader2Creator(")


@torchdata_available
@mark.parametrize("datapipe", DATAPIPES)
def test_dataloader_creator_create_dataset_object(datapipe: IterDataPipe | MapDataPipe) -> None:
    dl = DataLoader2(datapipe)
    dataloader = DataLoader2Creator(dl).create()
    assert isinstance(dataloader, DataLoader2)
    assert dataloader is dl
    assert tuple(dataloader) == (1, 2, 3, 4, 5)


@torchdata_available
@mark.parametrize("datapipe", DATAPIPES)
def test_dataloader_creator_create_dataset_config(datapipe: IterDataPipe | MapDataPipe) -> None:
    dataloader = DataLoader2Creator(
        {OBJECT_TARGET: "torchdata.dataloader2.DataLoader2", "datapipe": datapipe},
    ).create()
    assert isinstance(dataloader, DataLoader2)
    assert tuple(dataloader) == (1, 2, 3, 4, 5)


@torchdata_available
@mark.parametrize("datapipe", DATAPIPES)
def test_dataloader_creator_create_dataset_cache_true(datapipe: IterDataPipe | MapDataPipe) -> None:
    creator = DataLoader2Creator(
        {OBJECT_TARGET: "torchdata.dataloader2.DataLoader2", "datapipe": datapipe}, cache=True
    )
    dataloader1 = creator.create()
    dataloader2 = creator.create()
    assert isinstance(dataloader1, DataLoader2)
    assert isinstance(dataloader2, DataLoader2)
    assert dataloader1 is dataloader2
    assert objects_are_equal(tuple(dataloader1), tuple(dataloader2))


@torchdata_available
@mark.parametrize("datapipe", DATAPIPES)
def test_dataloader_creator_create_dataloader_cache_false(
    datapipe: IterDataPipe | MapDataPipe,
) -> None:
    creator = DataLoader2Creator(
        {OBJECT_TARGET: "torchdata.dataloader2.DataLoader2", "datapipe": datapipe},
    )
    dataloader1 = creator.create()
    dataloader2 = creator.create()
    assert isinstance(dataloader1, DataLoader2)
    assert isinstance(dataloader2, DataLoader2)
    assert dataloader1 is not dataloader2
    assert objects_are_equal(tuple(dataloader1), tuple(dataloader2))


@torchdata_available
@mark.parametrize("datapipe", DATAPIPES)
def test_dataloader_creator_create_with_engine(datapipe: IterDataPipe | MapDataPipe) -> None:
    dataloader = DataLoader2Creator(DataLoader2(datapipe)).create(engine=Mock(spec=BaseEngine))
    assert isinstance(dataloader, DataLoader2)
    assert tuple(dataloader) == (1, 2, 3, 4, 5)


###############################################
#     Tests for VanillaDataLoader2Creator     #
###############################################


@torchdata_available
def test_vanilla_dataloader2_creator_str() -> None:
    assert str(VanillaDataLoader2Creator(Mock(spec=IterDataPipe))).startswith(
        "VanillaDataLoader2Creator("
    )


@torchdata_available
@mark.parametrize("datapipe", DATAPIPES)
def test_vanilla_dataloader2_creator_datapipe(datapipe: IterDataPipe | MapDataPipe) -> None:
    dataloader = VanillaDataLoader2Creator(datapipe).create()
    assert isinstance(dataloader, DataLoader2)
    assert isinstance(dataloader.datapipe, (IterDataPipe, MapDataPipe))
    assert dataloader.datapipe_adapter_fns is None
    assert dataloader.reading_service is None
    assert tuple(dataloader) == (1, 2, 3, 4, 5)


@torchdata_available
def test_vanilla_dataloader2_creator_datapipe_creator() -> None:
    dataloader = VanillaDataLoader2Creator(
        ChainedDataPipeCreator(
            {
                OBJECT_TARGET: "torch.utils.data.datapipes.iter.IterableWrapper",
                "iterable": [1, 2, 3, 4, 5],
            }
        )
    ).create()
    assert isinstance(dataloader, DataLoader2)
    assert isinstance(dataloader.datapipe, IterableWrapper)
    assert dataloader.datapipe_adapter_fns is None
    assert dataloader.reading_service is None
    assert tuple(dataloader) == (1, 2, 3, 4, 5)


@torchdata_available
@mark.parametrize(
    "datapipe_adapter_fn",
    (
        Shuffle(),
        {OBJECT_TARGET: "torchdata.dataloader2.adapter.Shuffle"},
    ),
)
def test_vanilla_dataloader2_creator_datapipe_adapter_fn(
    datapipe: IterDataPipe, datapipe_adapter_fn: Adapter | dict
) -> None:
    dataloader = VanillaDataLoader2Creator(
        datapipe, datapipe_adapter_fn=datapipe_adapter_fn
    ).create()
    assert isinstance(dataloader, DataLoader2)
    assert isinstance(dataloader.datapipe, Shuffler)
    assert len(dataloader.datapipe_adapter_fns) == 1
    assert isinstance(dataloader.datapipe_adapter_fns[0], Shuffle)
    assert dataloader.reading_service is None
    assert set(dataloader) == {1, 2, 3, 4, 5}


@torchdata_available
@mark.parametrize(
    "reading_service",
    (
        MultiProcessingReadingService(),
        {OBJECT_TARGET: "torchdata.dataloader2.MultiProcessingReadingService"},
    ),
)
def test_vanilla_dataloader2_creator_reading_service(
    datapipe: IterDataPipe,
    reading_service: ReadingServiceInterface | dict,
) -> None:
    dataloader = VanillaDataLoader2Creator(datapipe, reading_service=reading_service).create()
    assert isinstance(dataloader, DataLoader2)
    assert isinstance(dataloader.datapipe, IterableWrapper)
    assert dataloader.datapipe_adapter_fns is None
    assert isinstance(dataloader.reading_service, MultiProcessingReadingService)
    assert tuple(dataloader) == (1, 2, 3, 4, 5)
