from unittest.mock import Mock, patch

from objectory import OBJECT_TARGET

from gravitorch.data.partitioners import (
    BasePartitioner,
    FixedSizePartitioner,
    setup_partitioner,
)

#######################################
#     Tests for setup_partitioner     #
#######################################


def test_setup_partitioner_object() -> None:
    creator = Mock(spec=BasePartitioner)
    assert setup_partitioner(creator) is creator


def test_setup_partitioner_dict_mock() -> None:
    creator_mock = Mock(factory=Mock(return_value="abc"))
    with patch("gravitorch.data.partitioners.base.BasePartitioner", creator_mock):
        assert setup_partitioner({OBJECT_TARGET: "name"}) == "abc"
        creator_mock.factory.assert_called_once_with(_target_="name")


def test_setup_partitioner_dict() -> None:
    assert isinstance(
        setup_partitioner(
            {
                OBJECT_TARGET: "gravitorch.data.partitioners.FixedSizePartitioner",
                "partition_size": 4,
            }
        ),
        FixedSizePartitioner,
    )
