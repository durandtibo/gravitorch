from objectory import OBJECT_TARGET

from gravitorch.loops.evaluation import VanillaEvaluationLoop, setup_evaluation_loop

###########################################
#     Tests for setup_evaluation_loop     #
###########################################


def test_setup_evaluation_loop_none() -> None:
    assert isinstance(setup_evaluation_loop(None), VanillaEvaluationLoop)


def test_setup_evaluation_loop_object() -> None:
    evaluation_loop = VanillaEvaluationLoop()
    assert setup_evaluation_loop(evaluation_loop) is evaluation_loop


def test_setup_evaluation_loop_dict() -> None:
    assert isinstance(
        setup_evaluation_loop({OBJECT_TARGET: "gravitorch.loops.evaluation.VanillaEvaluationLoop"}),
        VanillaEvaluationLoop,
    )
