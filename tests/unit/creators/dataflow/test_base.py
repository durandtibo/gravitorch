from __future__ import annotations

from objectory import OBJECT_TARGET

from gravitorch.creators.dataflow import (
    IterableDataFlowCreator,
    is_dataflow_creator_config,
    setup_dataflow_creator,
)

################################################
#     Tests for is_dataflow_creator_config     #
################################################


def test_is_dataflow_creator_config_true() -> None:
    assert is_dataflow_creator_config(
        {
            OBJECT_TARGET: "gravitorch.creators.dataflow.IterableDataFlowCreator",
            "iterable": (1, 2, 3, 4, 5),
        }
    )


def test_is_dataflow_creator_config_false() -> None:
    assert not is_dataflow_creator_config({"_target_": "torch.nn.Identity"})


############################################
#     Tests for setup_dataflow_creator     #
############################################


def test_setup_dataflow_creator_object() -> None:
    creator = IterableDataFlowCreator((1, 2, 3, 4, 5))
    assert setup_dataflow_creator(creator) is creator


def test_setup_dataflow_creator_dict() -> None:
    assert isinstance(
        setup_dataflow_creator(
            {
                OBJECT_TARGET: "gravitorch.creators.dataflow.IterableDataFlowCreator",
                "iterable": (1, 2, 3, 4, 5),
            }
        ),
        IterableDataFlowCreator,
    )
