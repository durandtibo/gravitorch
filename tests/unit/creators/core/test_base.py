from unittest.mock import Mock, patch

from objectory import OBJECT_TARGET

from gravitorch.creators.core import (
    BaseCoreCreator,
    is_core_creator_config,
    setup_core_creator,
)

############################################
#     Tests for is_core_creator_config     #
############################################


def test_is_core_creator_config_true() -> None:
    assert is_core_creator_config(
        {
            OBJECT_TARGET: "gravitorch.creators.core.VanillaCoreCreator",
            "datasource": {OBJECT_TARGET: "gravitorch.testing.DummyDataSource"},
            "model": {OBJECT_TARGET: "gravitorch.testing.DummyClassificationModel"},
            "optimizer": {OBJECT_TARGET: "torch.optim.SGD", "lr": 0.01},
            "lr_scheduler": {OBJECT_TARGET: "torch.optim.lr_scheduler.StepLR", "step_size": 5},
        }
    )


def test_is_core_creator_config_false() -> None:
    assert not is_core_creator_config({OBJECT_TARGET: "torch.nn.Identity"})


########################################
#     Tests for setup_core_creator     #
########################################


def test_setup_core_creator_object() -> None:
    creator = Mock(spec=BaseCoreCreator)
    assert setup_core_creator(creator) is creator


def test_setup_core_creator_dict_mock() -> None:
    creator_mock = Mock(factory=Mock(return_value="abc"))
    with patch("gravitorch.creators.core.base.BaseCoreCreator", creator_mock):
        assert setup_core_creator({OBJECT_TARGET: "name"}) == "abc"
        creator_mock.factory.assert_called_once_with(_target_="name")
