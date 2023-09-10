import logging

from objectory import OBJECT_TARGET
from pytest import LogCaptureFixture
from torch.nn import Identity

from gravitorch.loops.evaluation import (
    EvaluationLoop,
    is_evaluation_loop_config,
    setup_evaluation_loop,
)

###############################################
#     Tests for is_evaluation_loop_config     #
###############################################


def test_is_evaluation_loop_config_true() -> None:
    assert is_evaluation_loop_config({OBJECT_TARGET: "gravitorch.loops.evaluation.EvaluationLoop"})


def test_is_evaluation_loop_config_false() -> None:
    assert not is_evaluation_loop_config({OBJECT_TARGET: "torch.nn.Identity"})


###########################################
#     Tests for setup_evaluation_loop     #
###########################################


def test_setup_evaluation_loop_none() -> None:
    assert isinstance(setup_evaluation_loop(None), EvaluationLoop)


def test_setup_evaluation_loop_object() -> None:
    evaluation_loop = EvaluationLoop()
    assert setup_evaluation_loop(evaluation_loop) is evaluation_loop


def test_setup_evaluation_loop_dict() -> None:
    assert isinstance(
        setup_evaluation_loop({OBJECT_TARGET: "gravitorch.loops.evaluation.EvaluationLoop"}),
        EvaluationLoop,
    )


def test_setup_evaluation_loop_incorrect_type(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(level=logging.WARNING):
        assert isinstance(setup_evaluation_loop({OBJECT_TARGET: "torch.nn.Identity"}), Identity)
        assert caplog.messages
