from unittest.mock import Mock, patch

from objectory import OBJECT_TARGET

from gravitorch.creators.datapipe import (
    BaseIterDataPipeCreator,
    SequentialIterDataPipeCreator,
    setup_iterdatapipe_creator,
)

#################################################
#     Tests for setup_iterdatapipe_creator     #
#################################################


def test_setup_iterdatapipe_creator_object() -> None:
    creator = Mock(spec=BaseIterDataPipeCreator)
    assert setup_iterdatapipe_creator(creator) is creator


def test_setup_iterdatapipe_creator_dict_mock() -> None:
    creator_mock = Mock(factory=Mock(return_value="abc"))
    with patch("gravitorch.creators.datapipe.base.BaseIterDataPipeCreator", creator_mock):
        assert setup_iterdatapipe_creator({"_target_": "name"}) == "abc"
        creator_mock.factory.assert_called_once_with(_target_="name")


def test_setup_iterdatapipe_creator_dict() -> None:
    assert isinstance(
        setup_iterdatapipe_creator(
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
