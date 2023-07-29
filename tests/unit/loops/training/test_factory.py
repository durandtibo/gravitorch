import logging

from objectory import OBJECT_TARGET
from pytest import LogCaptureFixture
from torch.nn import Identity

from gravitorch.loops.training import (
    VanillaTrainingLoop,
    is_training_loop_config,
    setup_training_loop,
)

#############################################
#     Tests for is_training_loop_config     #
#############################################


def test_is_training_loop_config_true() -> None:
    assert is_training_loop_config({OBJECT_TARGET: "gravitorch.loops.training.VanillaTrainingLoop"})


def test_is_training_loop_config_false() -> None:
    assert not is_training_loop_config({OBJECT_TARGET: "torch.nn.Identity"})


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


def test_setup_training_loop_incorrect_type(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(level=logging.WARNING):
        assert isinstance(setup_training_loop({OBJECT_TARGET: "torch.nn.Identity"}), Identity)
        assert caplog.messages
