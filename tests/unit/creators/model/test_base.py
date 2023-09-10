import logging
from unittest.mock import Mock, patch

from objectory import OBJECT_TARGET
from pytest import LogCaptureFixture
from torch.nn import Module

from gravitorch.creators.model import (
    BaseModelCreator,
    ModelCreator,
    is_model_creator_config,
    setup_model_creator,
)

#############################################
#     Tests for is_model_creator_config     #
#############################################


def test_is_model_creator_config_true() -> None:
    assert is_model_creator_config(
        {
            OBJECT_TARGET: "gravitorch.creators.model.ModelCreator",
            "model_config": {"_target_": "gravitorch.testing.DummyClassificationModel"},
        }
    )


def test_is_model_creator_config_false() -> None:
    assert not is_model_creator_config({OBJECT_TARGET: "torch.nn.Identity"})


#########################################
#     Tests for setup_model_creator     #
#########################################


def test_setup_model_creator_object() -> None:
    model_creator = ModelCreator(model_config={})
    assert setup_model_creator(model_creator) is model_creator


def test_setup_model_creator_dict() -> None:
    assert isinstance(
        setup_model_creator(
            {
                OBJECT_TARGET: "gravitorch.creators.model.ModelCreator",
                "model_config": {},
            }
        ),
        ModelCreator,
    )


def test_setup_model_creator_dict_mock() -> None:
    factory_mock = Mock(return_value=Mock(spec=BaseModelCreator))
    with patch("gravitorch.creators.model.base.BaseModelCreator.factory", factory_mock):
        assert isinstance(setup_model_creator({OBJECT_TARGET: "name"}), BaseModelCreator)
        factory_mock.assert_called_once_with(_target_="name")


def test_setup_model_creator_incorrect_type(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(level=logging.WARNING):
        assert isinstance(setup_model_creator({OBJECT_TARGET: "torch.nn.Identity"}), Module)
        assert caplog.messages
