from objectory import OBJECT_TARGET

from gravitorch.creators.dataloader import (
    AutoDataLoaderCreator,
    DataLoaderCreator,
    setup_data_loader_creator,
)


def test_setup_data_loader_creator_none() -> None:
    assert isinstance(setup_data_loader_creator(None), AutoDataLoaderCreator)


def test_setup_data_loader_creator_config() -> None:
    data_loader_creator = setup_data_loader_creator(
        {OBJECT_TARGET: "gravitorch.creators.dataloader.DataLoaderCreator"}
    )
    assert isinstance(data_loader_creator, DataLoaderCreator)


def test_setup_data_loader_creator_object() -> None:
    data_loader_creator = setup_data_loader_creator(DataLoaderCreator())
    assert isinstance(data_loader_creator, DataLoaderCreator)
