from objectory import OBJECT_TARGET

from gravitorch.utils.evaluation_loops import (
    VanillaEvaluationLoop,
    setup_evaluation_loop,
)

###########################################
#     Tests for setup_evaluation_loop     #
###########################################


def test_setup_evaluation_loop_none():
    assert isinstance(setup_evaluation_loop(None), VanillaEvaluationLoop)


def test_setup_evaluation_loop_object():
    evaluation_loop = VanillaEvaluationLoop()
    assert setup_evaluation_loop(evaluation_loop) is evaluation_loop


def test_setup_evaluation_loop_dict():
    assert isinstance(
        setup_evaluation_loop(
            {OBJECT_TARGET: "gravitorch.utils.evaluation_loops.VanillaEvaluationLoop"}
        ),
        VanillaEvaluationLoop,
    )
