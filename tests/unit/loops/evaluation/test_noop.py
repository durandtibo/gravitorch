from unittest.mock import Mock

from gravitorch.engines import BaseEngine
from gravitorch.loops.evaluation import NoOpEvaluationLoop

########################################
#     Tests for NoOpEvaluationLoop     #
########################################


def test_noop_evaluation_loop_str():
    assert str(NoOpEvaluationLoop()).startswith("NoOpEvaluationLoop()")


def test_noop_evaluation_loop_eval():
    engine = Mock(spec=BaseEngine)
    NoOpEvaluationLoop().eval(engine)
    engine.assert_not_called()


def test_noop_evaluation_loop_load_state_dict():
    NoOpEvaluationLoop().load_state_dict({})  # Verify it does not raise error


def test_noop_evaluation_loop_state_dict():
    assert NoOpEvaluationLoop().state_dict() == {}
