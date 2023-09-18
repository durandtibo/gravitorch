from unittest.mock import Mock

from objectory import OBJECT_TARGET

from gravitorch.creators.dataloader2 import (
    DataLoader2Creator,
    is_dataloader2_creator_config,
    setup_dataloader2_creator,
)
from gravitorch.testing import torchdata_available
from gravitorch.utils.imports import is_torchdata_available

if is_torchdata_available():
    from torchdata.dataloader2 import DataLoader2

###################################################
#     Tests for is_dataloader2_creator_config     #
###################################################


@torchdata_available
def test_is_dataloader2_creator_config_true() -> None:
    assert is_dataloader2_creator_config(
        {
            OBJECT_TARGET: "gravitorch.creators.dataloader2.DataLoader2Creator",
            "dataloader": Mock(spec=DataLoader2),
        }
    )


@torchdata_available
def test_is_dataloader2_creator_config_false() -> None:
    assert not is_dataloader2_creator_config({"_target_": "torch.nn.Identity"})


###############################################
#     Tests for setup_dataloader2_creator     #
###############################################


@torchdata_available
def test_setup_dataloader2_creator_object() -> None:
    creator = DataLoader2Creator(Mock(spec=DataLoader2))
    assert setup_dataloader2_creator(creator) is creator


@torchdata_available
def test_setup_dataloader2_creator_dict() -> None:
    assert isinstance(
        setup_dataloader2_creator(
            {
                OBJECT_TARGET: "gravitorch.creators.dataloader2.DataLoader2Creator",
                "dataloader": Mock(spec=DataLoader2),
            }
        ),
        DataLoader2Creator,
    )
