import logging
from unittest.mock import Mock, patch

from objectory import OBJECT_TARGET
from pytest import LogCaptureFixture
from torch.nn import Module

from gravitorch.creators.datapipe import (
    BaseIterDataPipeCreator,
    SequentialIterDataPipeCreator,
    is_datapipe_creator_config,
    setup_iter_datapipe_creator,
)

################################################
#     Tests for is_datapipe_creator_config     #
################################################


def test_is_datapipe_creator_config_true() -> None:
    assert is_datapipe_creator_config(
        {
            OBJECT_TARGET: "gravitorch.creators.datapipe.SequentialIterDataPipeCreator",
            "config": [
                {
                    OBJECT_TARGET: "torch.utils.data.datapipes.iter.IterableWrapper",
                    "iterable": [1, 2, 3, 4],
                }
            ],
        }
    )


def test_is_datapipe_creator_config_false() -> None:
    assert not is_datapipe_creator_config({OBJECT_TARGET: "torch.nn.Identity"})


#################################################
#     Tests for setup_iter_datapipe_creator     #
#################################################


def test_setup_iter_datapipe_creator_object() -> None:
    creator = Mock(spec=BaseIterDataPipeCreator)
    assert setup_iter_datapipe_creator(creator) is creator


def test_setup_iter_datapipe_creator_dict() -> None:
    assert isinstance(
        setup_iter_datapipe_creator(
            {
                OBJECT_TARGET: "gravitorch.creators.datapipe.SequentialIterDataPipeCreator",
                "config": [
                    {
                        OBJECT_TARGET: "gravitorch.datapipes.iter.SourceWrapper",
                        "data": [1, 2, 3, 4],
                    }
                ],
            }
        ),
        SequentialIterDataPipeCreator,
    )


def test_setup_iter_datapipe_creator_dict_mock() -> None:
    factory_mock = Mock(return_value=Mock(spec=BaseIterDataPipeCreator))
    with patch("gravitorch.creators.datapipe.base.BaseIterDataPipeCreator.factory", factory_mock):
        assert isinstance(
            setup_iter_datapipe_creator({OBJECT_TARGET: "name"}), BaseIterDataPipeCreator
        )
        factory_mock.assert_called_once_with(_target_="name")


def test_setup_iter_datapipe_creator_incorrect_type(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(level=logging.WARNING):
        assert isinstance(setup_iter_datapipe_creator({OBJECT_TARGET: "torch.nn.Identity"}), Module)
        assert caplog.messages
