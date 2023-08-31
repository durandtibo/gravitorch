from unittest.mock import Mock

from minevent import EventHandler
from pytest import mark, raises

from gravitorch import constants as ct
from gravitorch.engines import BaseEngine, EngineEvents
from gravitorch.handlers import EarlyStopping
from gravitorch.utils.history import (
    EmptyHistoryError,
    GenericHistory,
    MaxScalarHistory,
    MinScalarHistory,
)

EVENTS = ("my_event", "my_other_event")
METRICS = ("metric1", "metric2")


###################################
#     Tests for EarlyStopping     #
###################################


def test_early_stopping_str() -> None:
    assert str(EarlyStopping()).startswith("EarlyStopping(")


@mark.parametrize("metric_name", METRICS)
def test_early_stopping_metric_name(metric_name: str) -> None:
    assert EarlyStopping(metric_name=metric_name)._metric_name == metric_name


@mark.parametrize("patience", (1, 5))
def test_early_stopping_patience(patience: int) -> None:
    assert EarlyStopping(patience=patience)._patience == patience


def test_early_stopping_incorrect_patience() -> None:
    with raises(ValueError, match="patience must be a positive integer"):
        EarlyStopping(patience=0)


@mark.parametrize("delta", (0.0, 1.0))
def test_early_stopping_delta(delta: float) -> None:
    assert EarlyStopping(delta=delta)._delta == delta


def test_early_stopping_incorrect_delta() -> None:
    with raises(ValueError, match="delta should not be a negative number"):
        EarlyStopping(delta=-0.1)


@mark.parametrize("cumulative_delta", (True, False))
def test_early_stopping_cumulative_delta(cumulative_delta: bool) -> None:
    assert EarlyStopping(cumulative_delta=cumulative_delta)._cumulative_delta == cumulative_delta


def test_early_stopping_attach_with_correct_metric() -> None:
    engine = Mock(
        spec=BaseEngine,
        has_event_handler=Mock(return_value=False),
        has_history=Mock(return_value=True),
        get_history=Mock(return_value=MinScalarHistory("my_metric")),
    )
    handler = EarlyStopping(metric_name="my_metric")
    handler.attach(engine)
    assert engine.add_event_handler.call_args_list[0].args == (
        EngineEvents.TRAIN_STARTED,
        EventHandler(handler.start, handler_kwargs={"engine": engine}),
    )
    assert engine.add_event_handler.call_args_list[1].args == (
        EngineEvents.EPOCH_COMPLETED,
        EventHandler(handler.step, handler_kwargs={"engine": engine}),
    )
    engine.add_module.assert_called_once_with(ct.EARLY_STOPPING, handler)


def test_early_stopping_attach_with_incorrect_metric() -> None:
    engine = Mock(
        spec=BaseEngine,
        has_history=Mock(return_value=True),
        get_history=Mock(return_value=GenericHistory("my_metric")),
    )
    handler = EarlyStopping(metric_name="my_metric")
    with raises(
        RuntimeError,
        match=(
            "The early stopping handler only supports ``MaxScalarHistory`` or "
            "``MinScalarHistory`` history tracker"
        ),
    ):
        handler.attach(engine)


def test_early_stopping_attach_without_metric() -> None:
    engine = Mock(
        spec=BaseEngine,
        has_event_handler=Mock(return_value=False),
        has_history=Mock(return_value=False),
    )
    handler = EarlyStopping()
    handler.attach(engine)
    assert engine.add_event_handler.call_args_list[0].args == (
        EngineEvents.TRAIN_STARTED,
        EventHandler(handler.start, handler_kwargs={"engine": engine}),
    )
    assert engine.add_event_handler.call_args_list[1].args == (
        EngineEvents.EPOCH_COMPLETED,
        EventHandler(handler.step, handler_kwargs={"engine": engine}),
    )
    engine.add_module.assert_called_once_with(ct.EARLY_STOPPING, handler)


def test_early_stopping_load_state_dict() -> None:
    handler = EarlyStopping()
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


def test_early_stopping_state_dict() -> None:
    handler = EarlyStopping()
    assert handler.state_dict() == {
        "best_epoch": None,
        "best_score": None,
        "waiting_counter": 0,
    }


def test_early_stopping_start_should_terminate_false() -> None:
    engine = Mock(spec=BaseEngine)
    handler = EarlyStopping()
    handler.start(engine)
    engine.terminate.assert_not_called()


def test_early_stopping_start_should_terminate_true() -> None:
    engine = Mock(spec=BaseEngine)
    handler = EarlyStopping()
    handler.load_state_dict(
        {
            "best_epoch": 12,
            "best_score": 5,
            "waiting_counter": 5,
        }
    )
    handler.start(engine)
    engine.terminate.assert_called_once()


def test_early_stopping_step_empty_history() -> None:
    engine = Mock(spec=BaseEngine, get_history=Mock(return_value=MinScalarHistory("my_metric")))
    handler = EarlyStopping(metric_name="my_metric")
    with raises(EmptyHistoryError, match="'my_metric' history is empty."):
        handler.step(engine)


def test_early_stopping_step_first_epoch() -> None:
    handler = EarlyStopping(metric_name="my_metric")
    engine = Mock(
        spec=BaseEngine,
        epoch=0,
        get_history=Mock(return_value=MinScalarHistory("my_metric", elements=((None, 32),))),
    )
    handler.step(engine)
    assert handler._best_epoch == 0
    assert handler._best_score == 32
    assert handler._waiting_counter == 0


def test_early_stopping_step_reset_waiting_counter() -> None:
    handler = EarlyStopping(metric_name="my_metric")
    handler.load_state_dict(
        {
            "best_epoch": 12,
            "best_score": 5,
            "waiting_counter": 3,
        }
    )
    engine = Mock(
        spec=BaseEngine,
        epoch=16,
        get_history=Mock(return_value=MinScalarHistory("my_metric", elements=((None, 4),))),
    )
    handler.step(engine)
    assert handler._best_epoch == 16
    assert handler._best_score == 4
    assert handler._waiting_counter == 0


def test_early_stopping_step_increase_waiting_counter() -> None:
    handler = EarlyStopping(metric_name="my_metric")
    handler.load_state_dict(
        {
            "best_epoch": 12,
            "best_score": 5,
            "waiting_counter": 3,
        }
    )
    engine = Mock(
        spec=BaseEngine,
        epoch=16,
        get_history=Mock(return_value=MinScalarHistory("my_metric", elements=((None, 6),))),
    )
    handler.step(engine)
    assert handler._best_epoch == 12
    assert handler._best_score == 5
    assert handler._waiting_counter == 4


def test_early_stopping_step_waiting_counter_reach_patience() -> None:
    handler = EarlyStopping(metric_name="my_metric")
    handler.load_state_dict(
        {
            "best_epoch": 12,
            "best_score": 5,
            "waiting_counter": 4,
        }
    )
    engine = Mock(
        spec=BaseEngine,
        get_history=Mock(return_value=MinScalarHistory("my_metric", elements=((None, 6),))),
    )
    handler.step(engine)
    engine.terminate.assert_called_once()
    assert handler._best_epoch == 12
    assert handler._best_score == 5
    assert handler._waiting_counter == 5


def test_early_stopping_step_increase_waiting_counter_delta_1_min() -> None:
    handler = EarlyStopping(metric_name="my_metric", delta=1)
    handler.load_state_dict(
        {
            "best_epoch": 12,
            "best_score": 5,
            "waiting_counter": 3,
        }
    )
    engine = Mock(
        spec=BaseEngine,
        epoch=16,
        get_history=Mock(return_value=MinScalarHistory("my_metric", elements=((None, 4.5),))),
    )
    handler.step(engine)
    assert handler._best_epoch == 12
    assert handler._best_score == 4.5
    assert handler._waiting_counter == 4


def test_early_stopping_step_increase_waiting_counter_delta_1_max() -> None:
    handler = EarlyStopping(metric_name="my_metric", delta=1)
    handler.load_state_dict(
        {
            "best_epoch": 12,
            "best_score": 5,
            "waiting_counter": 3,
        }
    )
    engine = Mock(
        spec=BaseEngine,
        epoch=16,
        get_history=Mock(return_value=MaxScalarHistory("my_metric", elements=((None, 5.5),))),
    )
    handler.step(engine)
    assert handler._best_epoch == 12
    assert handler._best_score == 5.5
    assert handler._waiting_counter == 4


def test_early_stopping_step_increase_waiting_counter_cumulative_delta_true_min() -> None:
    handler = EarlyStopping(metric_name="my_metric", delta=1, cumulative_delta=True)
    handler.load_state_dict(
        {
            "best_epoch": 12,
            "best_score": 5,
            "waiting_counter": 3,
        }
    )
    engine = Mock(
        spec=BaseEngine,
        epoch=16,
        get_history=Mock(return_value=MinScalarHistory("my_metric", elements=((None, 4.5),))),
    )
    handler.step(engine)
    assert handler._best_epoch == 12
    assert handler._best_score == 5
    assert handler._waiting_counter == 4


def test_early_stopping_step_increase_waiting_counter_cumulative_delta_true_max() -> None:
    handler = EarlyStopping(metric_name="my_metric", delta=1, cumulative_delta=True)
    handler.load_state_dict(
        {
            "best_epoch": 12,
            "best_score": 5,
            "waiting_counter": 3,
        }
    )
    engine = Mock(
        spec=BaseEngine,
        epoch=16,
        get_history=Mock(return_value=MaxScalarHistory("my_metric", elements=((None, 5.5),))),
    )
    handler.step(engine)
    assert handler._best_epoch == 12
    assert handler._best_score == 5
    assert handler._waiting_counter == 4
