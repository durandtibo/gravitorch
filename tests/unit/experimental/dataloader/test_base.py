from unittest.mock import Mock

from objectory import OBJECT_TARGET
from torch.utils.data import DataLoader

from gravitorch.creators.dataloader import (
    DataLoaderCreator,
    is_dataloader_creator_config,
    setup_dataloader_creator,
)

##################################################
#     Tests for is_dataloader_creator_config     #
##################################################


def test_is_dataloader_creator_config_true() -> None:
    assert is_dataloader_creator_config(
        {
            OBJECT_TARGET: "gravitorch.creators.dataloader.DataLoaderCreator",
            "dataloader": Mock(spec=DataLoader),
        }
    )


def test_is_dataloader_creator_config_false() -> None:
    assert not is_dataloader_creator_config({"_target_": "torch.nn.Identity"})


##############################################
#     Tests for setup_dataloader_creator     #
##############################################


def test_setup_dataloader_creator_object() -> None:
    creator = DataLoaderCreator(Mock(spec=DataLoader))
    assert setup_dataloader_creator(creator) is creator


def test_setup_dataloader_creator_dict() -> None:
    assert isinstance(
        setup_dataloader_creator(
            {
                OBJECT_TARGET: "gravitorch.creators.dataloader.DataLoaderCreator",
                "dataloader": Mock(spec=DataLoader),
            }
        ),
        DataLoaderCreator,
    )
