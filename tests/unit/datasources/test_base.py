from __future__ import annotations

import logging
from unittest.mock import Mock, patch

from objectory import OBJECT_TARGET
from pytest import LogCaptureFixture
from torch.nn import Identity

from gravitorch.datasources import (
    BaseDataSource,
    IterDataPipeCreatorDataSource,
    is_datasource_config,
    setup_and_attach_datasource,
    setup_datasource,
)
from gravitorch.engines import BaseEngine

##########################################
#     Tests for is_datasource_config     #
##########################################


def test_is_datasource_config_true() -> None:
    assert is_datasource_config(
        {OBJECT_TARGET: "gravitorch.datasources.IterDataPipeCreatorDataSource"}
    )


def test_is_datasource_config_false() -> None:
    assert not is_datasource_config({OBJECT_TARGET: "torch.nn.Identity"})


######################################
#     Tests for setup_datasource     #
######################################


def test_setup_datasource_object() -> None:
    source = Mock(spec=BaseDataSource)
    assert setup_datasource(source) is source


def test_setup_datasource_dict_mock() -> None:
    factory_mock = Mock(return_value="abc")
    with patch("gravitorch.datasources.base.BaseDataSource.factory", factory_mock):
        assert setup_datasource({OBJECT_TARGET: "name"}) == "abc"
        factory_mock.assert_called_once_with(_target_="name")


def test_setup_datasource_dict() -> None:
    assert isinstance(
        setup_datasource(
            {
                OBJECT_TARGET: "gravitorch.datasources.IterDataPipeCreatorDataSource",
                "datapipe_creators": {
                    "train": {
                        OBJECT_TARGET: "gravitorch.creators.datapipe.ChainedDataPipeCreator",
                        "config": [
                            {
                                OBJECT_TARGET: "torch.utils.data.datapipes.iter.IterableWrapper",
                                "data": [1, 2, 3, 4],
                            }
                        ],
                    },
                },
            }
        ),
        IterDataPipeCreatorDataSource,
    )


def test_setup_datasource_incorrect_type(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(level=logging.WARNING):
        assert isinstance(setup_datasource({OBJECT_TARGET: "torch.nn.Identity"}), Identity)
        assert caplog.messages


##################################################
#     Tests for setup_and_attach_datasource     #
##################################################


def test_setup_and_attach_datasource() -> None:
    engine = Mock(spec=BaseEngine)
    source = Mock(spec=BaseDataSource)
    assert setup_and_attach_datasource(source, engine) is source
    source.attach.assert_called_once_with(engine)
