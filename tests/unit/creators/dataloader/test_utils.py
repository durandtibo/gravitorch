from objectory import OBJECT_TARGET

from gravitorch.creators.dataloader import (
    AutoDataLoaderCreator,
    VanillaDataLoaderCreator,
    setup_data_loader_creator,
)


def test_setup_data_loader_creator_none():
    assert isinstance(setup_data_loader_creator(None), AutoDataLoaderCreator)


def test_setup_data_loader_creator_config():
    data_loader_creator = setup_data_loader_creator(
        {OBJECT_TARGET: "gravitorch.creators.dataloader.VanillaDataLoaderCreator"}
    )
    assert isinstance(data_loader_creator, VanillaDataLoaderCreator)


def test_setup_data_loader_creator_object():
    data_loader_creator = setup_data_loader_creator(VanillaDataLoaderCreator())
    assert isinstance(data_loader_creator, VanillaDataLoaderCreator)
