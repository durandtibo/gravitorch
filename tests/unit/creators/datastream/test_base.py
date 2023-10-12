from __future__ import annotations

from objectory import OBJECT_TARGET

from gravitorch.creators.datastream import (
    IterableDataStreamCreator,
    is_datastream_creator_config,
    setup_datastream_creator,
)

################################################
#     Tests for is_datastream_creator_config     #
################################################


def test_is_datastream_creator_config_true() -> None:
    assert is_datastream_creator_config(
        {
            OBJECT_TARGET: "gravitorch.creators.datastream.IterableDataFlowCreator",
            "iterable": (1, 2, 3, 4, 5),
        }
    )


def test_is_datastream_creator_config_false() -> None:
    assert not is_datastream_creator_config({"_target_": "torch.nn.Identity"})


############################################
#     Tests for setup_datastream_creator     #
############################################


def test_setup_datastream_creator_object() -> None:
    creator = IterableDataStreamCreator((1, 2, 3, 4, 5))
    assert setup_datastream_creator(creator) is creator


def test_setup_datastream_creator_dict() -> None:
    assert isinstance(
        setup_datastream_creator(
            {
                OBJECT_TARGET: "gravitorch.creators.datastream.IterableDataFlowCreator",
                "iterable": (1, 2, 3, 4, 5),
            }
        ),
        IterableDataStreamCreator,
    )
