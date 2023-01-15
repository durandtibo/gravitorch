from objectory import OBJECT_TARGET

from gravitorch.utils.training_loops import VanillaTrainingLoop, setup_training_loop

#########################################
#     Tests for setup_training_loop     #
#########################################


def test_setup_training_loop_none():
    assert isinstance(setup_training_loop(None), VanillaTrainingLoop)


def test_setup_training_loop_object():
    training_loop = VanillaTrainingLoop()
    assert setup_training_loop(training_loop) is training_loop


def test_setup_training_loop_dict():
    assert isinstance(
        setup_training_loop({OBJECT_TARGET: "gravitorch.utils.training_loops.VanillaTrainingLoop"}),
        VanillaTrainingLoop,
    )
