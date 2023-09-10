import logging

from objectory import OBJECT_TARGET
from pytest import LogCaptureFixture
from torch.nn import Module

from gravitorch.creators.dataloader import (
    AutoDataLoaderCreator,
    DataLoaderCreator,
    is_dataloader_creator_config,
    setup_dataloader_creator,
)

##################################################
#     Tests for is_dataloader_creator_config     #
##################################################


def test_is_dataloader_creator_config_true() -> None:
    assert is_dataloader_creator_config(
        {OBJECT_TARGET: "gravitorch.creators.dataloader.DataLoaderCreator"}
    )


def test_is_dataloader_creator_config_false() -> None:
    assert not is_dataloader_creator_config({OBJECT_TARGET: "torch.nn.Identity"})


##############################################
#     Tests for setup_dataloader_creator     #
##############################################


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


def test_setup_dataloader_creator_incorrect_type(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(level=logging.WARNING):
        assert isinstance(setup_dataloader_creator({OBJECT_TARGET: "torch.nn.Identity"}), Module)
        assert caplog.messages
