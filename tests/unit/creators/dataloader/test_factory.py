from objectory import OBJECT_TARGET

from gravitorch.creators.dataloader import (
    AutoDataLoaderCreator,
    DataLoaderCreator,
    setup_dataloader_creator,
)


def test_setup_dataloader_creator_none() -> None:
    assert isinstance(setup_dataloader_creator(None), AutoDataLoaderCreator)


def test_setup_dataloader_creator_config() -> None:
    dataloader_creator = setup_dataloader_creator(
        {OBJECT_TARGET: "gravitorch.creators.dataloader.DataLoaderCreator"}
    )
    assert isinstance(dataloader_creator, DataLoaderCreator)


def test_setup_dataloader_creator_object() -> None:
    dataloader_creator = setup_dataloader_creator(DataLoaderCreator())
    assert isinstance(dataloader_creator, DataLoaderCreator)
