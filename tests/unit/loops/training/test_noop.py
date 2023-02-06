from unittest.mock import Mock

from gravitorch.engines import BaseEngine
from gravitorch.loops.training import NoOpTrainingLoop

######################################
#     Tests for NoOpTrainingLoop     #
######################################


def test_noop_training_loop_str():
    assert str(NoOpTrainingLoop()).startswith("NoOpTrainingLoop()")


def test_noop_training_loop_train():
    engine = Mock(spec=BaseEngine)
    NoOpTrainingLoop().train(engine)
    engine.assert_not_called()


def test_noop_training_loop_load_state_dict():
    NoOpTrainingLoop().load_state_dict({})  # Verify it does not raise error


def test_noop_training_loop_state_dict():
    assert NoOpTrainingLoop().state_dict() == {}
