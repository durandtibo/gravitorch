from unittest.mock import Mock, patch

from objectory import OBJECT_TARGET

from gravitorch.creators.datapipe import (
    BaseIterDataPipeCreator,
    SequentialIterDataPipeCreator,
    setup_iter_datapipe_creator,
)

#################################################
#     Tests for setup_iter_datapipe_creator     #
#################################################


def test_setup_iter_datapipe_creator_object():
    creator = Mock(spec=BaseIterDataPipeCreator)
    assert setup_iter_datapipe_creator(creator) is creator


def test_setup_iter_datapipe_creator_dict_mock():
    creator_mock = Mock(factory=Mock(return_value="abc"))
    with patch("gravitorch.creators.datapipe.base.BaseIterDataPipeCreator", creator_mock):
        assert setup_iter_datapipe_creator({"_target_": "name"}) == "abc"
        creator_mock.factory.assert_called_once_with(_target_="name")


def test_setup_iter_datapipe_creator_dict():
    assert isinstance(
        setup_iter_datapipe_creator(
            {
                OBJECT_TARGET: "gravitorch.creators.datapipe.SequentialIterDataPipeCreator",
                "config": [
                    {
                        OBJECT_TARGET: "gravitorch.data.datapipes.iter.SourceIterDataPipe",
                        "data": [1, 2, 3, 4],
                    }
                ],
            }
        ),
        SequentialIterDataPipeCreator,
    )
