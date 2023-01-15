from unittest.mock import Mock

from pytest import mark, raises

from gravitorch import constants as ct
from gravitorch.engines import EngineEvents
from gravitorch.handlers import EarlyStoppingHandler
from gravitorch.utils.events import VanillaEventHandler
from gravitorch.utils.history import (
    EmptyHistoryError,
    GenericHistory,
    MaxScalarHistory,
    MinScalarHistory,
)

EVENTS = ("my_event", "my_other_event")
METRICS = ("metric1", "metric2")


##########################################
#     Tests for EarlyStoppingHandler     #
##########################################


def test_early_stopping_str():
    assert str(EarlyStoppingHandler()).startswith("EarlyStoppingHandler(")


@mark.parametrize("metric_name", METRICS)
def test_early_stopping_metric_name(metric_name: str):
    assert EarlyStoppingHandler(metric_name=metric_name)._metric_name == metric_name


@mark.parametrize("patience", (1, 5))
def test_early_stopping_patience(patience: int):
    assert EarlyStoppingHandler(patience=patience)._patience == patience


def test_early_stopping_incorrect_patience():
    with raises(ValueError):
        EarlyStoppingHandler(patience=0)


@mark.parametrize("delta", (0.0, 1.0))
def test_early_stopping_delta(delta: float):
    assert EarlyStoppingHandler(delta=delta)._delta == delta


def test_early_stopping_incorrect_delta():
    with raises(ValueError):
        EarlyStoppingHandler(delta=-0.1)


@mark.parametrize("cumulative_delta", (True, False))
def test_early_stopping_cumulative_delta(cumulative_delta: bool):
    assert (
        EarlyStoppingHandler(cumulative_delta=cumulative_delta)._cumulative_delta
        == cumulative_delta
    )


def test_early_stopping_attach_with_correct_metric():
    engine = Mock()
    engine.has_event_handler.return_value = False
    engine.has_history.return_value = True
    engine.get_history.return_value = MinScalarHistory("my_metric")
    handler = EarlyStoppingHandler(metric_name="my_metric")
    handler.attach(engine)
    assert engine.add_event_handler.call_args_list[0].args == (
        EngineEvents.TRAIN_STARTED,
        VanillaEventHandler(handler.start, handler_kwargs={"engine": engine}),
    )
    assert engine.add_event_handler.call_args_list[1].args == (
        EngineEvents.EPOCH_COMPLETED,
        VanillaEventHandler(handler.step, handler_kwargs={"engine": engine}),
    )
    engine.add_module.assert_called_once_with(ct.EARLY_STOPPING, handler)


def test_early_stopping_attach_with_incorrect_metric():
    engine = Mock()
    engine.has_history.return_value = True
    engine.get_history.return_value = GenericHistory("my_metric")
    handler = EarlyStoppingHandler(metric_name="my_metric")
    with raises(RuntimeError):
        handler.attach(engine)


def test_early_stopping_attach_without_metric():
    engine = Mock()
    handler = EarlyStoppingHandler()
    engine.has_event_handler.return_value = False
    engine.has_history.return_value = False
    handler.attach(engine)
    assert engine.add_event_handler.call_args_list[0].args == (
        EngineEvents.TRAIN_STARTED,
        VanillaEventHandler(handler.start, handler_kwargs={"engine": engine}),
    )
    assert engine.add_event_handler.call_args_list[1].args == (
        EngineEvents.EPOCH_COMPLETED,
        VanillaEventHandler(handler.step, handler_kwargs={"engine": engine}),
    )
    engine.add_module.assert_called_once_with(ct.EARLY_STOPPING, handler)


def test_early_stopping_load_state_dict():
    handler = EarlyStoppingHandler()
    handler.load_state_dict(
        {
            "best_epoch": 12,
            "best_score": 5,
            "waiting_counter": 3,
        }
    )
    assert handler._best_epoch == 12
    assert handler._best_score == 5
    assert handler._waiting_counter == 3


def test_early_stopping_state_dict():
    handler = EarlyStoppingHandler()
    assert handler.state_dict() == {
        "best_epoch": None,
        "best_score": None,
        "waiting_counter": 0,
    }


def test_early_stopping_start_should_terminate_false():
    engine = Mock()
    handler = EarlyStoppingHandler()
    handler.start(engine)
    engine.terminate.assert_not_called()


def test_early_stopping_start_should_terminate_true():
    engine = Mock()
    handler = EarlyStoppingHandler()
    handler.load_state_dict(
        {
            "best_epoch": 12,
            "best_score": 5,
            "waiting_counter": 5,
        }
    )
    handler.start(engine)
    engine.terminate.assert_called_once()


def test_early_stopping_step_empty_history():
    engine = Mock()
    engine.get_history.return_value = MinScalarHistory("my_metric")
    handler = EarlyStoppingHandler(metric_name="my_metric")
    with raises(EmptyHistoryError):
        handler.step(engine)


def test_early_stopping_step_first_epoch():
    handler = EarlyStoppingHandler(metric_name="my_metric")
    engine = Mock()
    engine.epoch = 0
    history = MinScalarHistory("my_metric")
    history.add_value(32)
    engine.get_history.return_value = history
    handler.step(engine)
    assert handler._best_epoch == 0
    assert handler._best_score == 32
    assert handler._waiting_counter == 0


def test_early_stopping_step_reset_waiting_counter():
    handler = EarlyStoppingHandler(metric_name="my_metric")
    handler.load_state_dict(
        {
            "best_epoch": 12,
            "best_score": 5,
            "waiting_counter": 3,
        }
    )
    engine = Mock()
    engine.epoch = 16
    history = MinScalarHistory("my_metric")
    history.add_value(4)
    engine.get_history.return_value = history
    handler.step(engine)
    assert handler._best_epoch == 16
    assert handler._best_score == 4
    assert handler._waiting_counter == 0


def test_early_stopping_step_increase_waiting_counter():
    handler = EarlyStoppingHandler(metric_name="my_metric")
    handler.load_state_dict(
        {
            "best_epoch": 12,
            "best_score": 5,
            "waiting_counter": 3,
        }
    )
    engine = Mock()
    engine.epoch = 16
    history = MinScalarHistory("my_metric")
    history.add_value(6)
    engine.get_history.return_value = history
    handler.step(engine)
    assert handler._best_epoch == 12
    assert handler._best_score == 5
    assert handler._waiting_counter == 4


def test_early_stopping_step_waiting_counter_reach_patience():
    handler = EarlyStoppingHandler(metric_name="my_metric")
    handler.load_state_dict(
        {
            "best_epoch": 12,
            "best_score": 5,
            "waiting_counter": 4,
        }
    )
    engine = Mock()
    history = MinScalarHistory("my_metric")
    history.add_value(6)
    engine.get_history.return_value = history
    handler.step(engine)
    engine.terminate.assert_called_once()
    assert handler._best_epoch == 12
    assert handler._best_score == 5
    assert handler._waiting_counter == 5


def test_early_stopping_step_increase_waiting_counter_delta_1_min():
    handler = EarlyStoppingHandler(metric_name="my_metric", delta=1)
    handler.load_state_dict(
        {
            "best_epoch": 12,
            "best_score": 5,
            "waiting_counter": 3,
        }
    )
    engine = Mock()
    engine.epoch = 16
    history = MinScalarHistory("my_metric")
    history.add_value(4.5)
    engine.get_history.return_value = history
    handler.step(engine)
    assert handler._best_epoch == 12
    assert handler._best_score == 4.5
    assert handler._waiting_counter == 4


def test_early_stopping_step_increase_waiting_counter_delta_1_max():
    handler = EarlyStoppingHandler(metric_name="my_metric", delta=1)
    handler.load_state_dict(
        {
            "best_epoch": 12,
            "best_score": 5,
            "waiting_counter": 3,
        }
    )
    engine = Mock()
    engine.epoch = 16
    history = MaxScalarHistory("my_metric")
    history.add_value(5.5)
    engine.get_history.return_value = history
    handler.step(engine)
    assert handler._best_epoch == 12
    assert handler._best_score == 5.5
    assert handler._waiting_counter == 4


def test_early_stopping_step_increase_waiting_counter_cumulative_delta_true_min():
    handler = EarlyStoppingHandler(metric_name="my_metric", delta=1, cumulative_delta=True)
    handler.load_state_dict(
        {
            "best_epoch": 12,
            "best_score": 5,
            "waiting_counter": 3,
        }
    )
    engine = Mock()
    engine.epoch = 16
    history = MinScalarHistory("my_metric")
    history.add_value(4.5)
    engine.get_history.return_value = history
    handler.step(engine)
    assert handler._best_epoch == 12
    assert handler._best_score == 5
    assert handler._waiting_counter == 4


def test_early_stopping_step_increase_waiting_counter_cumulative_delta_true_max():
    handler = EarlyStoppingHandler(metric_name="my_metric", delta=1, cumulative_delta=True)
    handler.load_state_dict(
        {
            "best_epoch": 12,
            "best_score": 5,
            "waiting_counter": 3,
        }
    )
    engine = Mock()
    engine.epoch = 16
    history = MaxScalarHistory("my_metric")
    history.add_value(5.5)
    engine.get_history.return_value = history
    handler.step(engine)
    assert handler._best_epoch == 12
    assert handler._best_score == 5
    assert handler._waiting_counter == 4
