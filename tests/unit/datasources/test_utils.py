from unittest.mock import Mock, patch

from objectory import OBJECT_TARGET

from gravitorch.datasources import (
    BaseDataSource,
    IterDataPipeCreatorDataSource,
    setup_and_attach_data_source,
    setup_data_source,
)
from gravitorch.engines import BaseEngine

#######################################
#     Tests for setup_data_source     #
#######################################


def test_setup_data_source_object():
    source = Mock(spec=BaseDataSource)
    assert setup_data_source(source) is source


def test_setup_data_source_dict_mock():
    source_mock = Mock(factory=Mock(return_value="abc"))
    with patch("gravitorch.datasources.utils.BaseDataSource", source_mock):
        assert setup_data_source({"_target_": "name"}) == "abc"
        source_mock.factory.assert_called_once_with(_target_="name")


def test_setup_data_source_dict():
    assert isinstance(
        setup_data_source(
            {
                "_target_": "gravitorch.datasources.IterDataPipeCreatorDataSource",
                "datapipe_creators": {
                    "train": {
                        OBJECT_TARGET: "gravitorch.creators.datapipe.SequentialIterDataPipeCreator",
                        "config": [
                            {
                                OBJECT_TARGET: "gravitorch.data.datapipes.iter.SourceIterDataPipe",
                                "data": [1, 2, 3, 4],
                            }
                        ],
                    },
                },
            }
        ),
        IterDataPipeCreatorDataSource,
    )


##################################################
#     Tests for setup_and_attach_data_source     #
##################################################


def test_setup_and_attach_data_source():
    engine = Mock(spec=BaseEngine)
    source = Mock(spec=BaseDataSource)
    assert setup_and_attach_data_source(source, engine) is source
    source.attach.assert_called_once_with(engine)
