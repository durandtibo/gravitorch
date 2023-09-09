from unittest.mock import Mock, patch

from objectory import OBJECT_TARGET

from gravitorch.data.partitioners import (
    BasePartitioner,
    FixedSizePartitioner,
    is_partitioner_config,
    setup_partitioner,
)

###########################################
#     Tests for is_partitioner_config     #
###########################################`


def test_is_partitioner_config_true() -> None:
    assert is_partitioner_config(
        {
            OBJECT_TARGET: "gravitorch.data.partitioners.FixedSizePartitioner",
            "partition_size": 3,
        }
    )


def test_is_partitioner_config_false() -> None:
    assert not is_partitioner_config({OBJECT_TARGET: "torch.nn.Identity"})


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
