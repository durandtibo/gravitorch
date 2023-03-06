from objectory import OBJECT_TARGET

from gravitorch.loops.training import VanillaTrainingLoop, setup_training_loop

#########################################
#     Tests for setup_training_loop     #
#########################################


def test_setup_training_loop_none() -> None:
    assert isinstance(setup_training_loop(None), VanillaTrainingLoop)


def test_setup_training_loop_object() -> None:
    training_loop = VanillaTrainingLoop()
    assert setup_training_loop(training_loop) is training_loop


def test_setup_training_loop_dict() -> None:
    assert isinstance(
        setup_training_loop({OBJECT_TARGET: "gravitorch.loops.training.VanillaTrainingLoop"}),
        VanillaTrainingLoop,
    )
