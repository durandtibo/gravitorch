from __future__ import annotations

from torch.utils.data import DataLoader

from gravitorch.creators.datastream import DataLoaderDataStreamCreator
from gravitorch.datasets import ExampleDataset
from gravitorch.datastreams import DataLoaderDataStream
from gravitorch.experimental.dataloader import VanillaDataLoaderCreator

###############################################
#     Tests for DataLoaderDataStreamCreator     #
###############################################


def test_dataloader_datastream_creator_str() -> None:
    assert str(DataLoaderDataStreamCreator(DataLoader(ExampleDataset((1, 2, 3, 4, 5))))).startswith(
        "DataLoaderDataStreamCreator("
    )


def test_dataloader_datastream_creator_create_dataloader() -> None:
    datastream = DataLoaderDataStreamCreator(DataLoader(ExampleDataset((1, 2, 3, 4, 5)))).create()
    assert isinstance(datastream, DataLoaderDataStream)
    assert list(datastream) == [1, 2, 3, 4, 5]


def test_dataloader_datastream_creator_create_dataloader_creator() -> None:
    datastream = DataLoaderDataStreamCreator(
        VanillaDataLoaderCreator(ExampleDataset((1, 2, 3, 4, 5)))
    ).create()
    assert isinstance(datastream, DataLoaderDataStream)
    assert list(datastream) == [1, 2, 3, 4, 5]
