from unittest.mock import Mock

from gravitorch import constants as ct
from gravitorch.engines import AlphaEngine, BaseEngine, EngineEvents
from gravitorch.handlers import EarlyStopping
from gravitorch.utils.engine_states import VanillaEngineState
from gravitorch.utils.history import MaxScalarHistory, MinScalarHistory

EVENTS = ("my_event", "my_other_event")
METRICS = ("metric1", "metric2")


def create_engine(**kwargs) -> BaseEngine:
    creator = Mock()
    creator.create.return_value = (None, None, None, None)
    return AlphaEngine(creator, **kwargs)


##########################################
#     Tests for EarlyStopping     #
##########################################


def test_early_stopping_simulation_min():
    engine = create_engine()
    engine.add_history(MinScalarHistory(f"{ct.EVAL}/loss"))
    handler = EarlyStopping(delta=0)
    handler.attach(engine)
    # Simulate the behavior of the engine.
    for loss in [3, 2, 1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]:
        engine.increment_epoch()
        engine.log_metric(f"{ct.EVAL}/loss", loss)
        engine.fire_event(EngineEvents.EPOCH_COMPLETED)
        if engine.should_terminate:
            break
    assert engine.epoch == 12
    assert handler.state_dict() == {
        "waiting_counter": 5,
        "best_epoch": 7,
        "best_score": 0.5,
    }


def test_early_stopping_simulation_max():
    engine = create_engine()
    engine.add_history(MaxScalarHistory("accuracy"))
    handler = EarlyStopping(metric_name="accuracy", delta=0)
    handler.attach(engine)
    # Simulate the behavior of the engine.
    for loss in [1, 2, 3, 3.1, 3.2, 3.3, 3.4, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5]:
        engine.increment_epoch()
        engine.log_metric("accuracy", loss)
        engine.fire_event(EngineEvents.EPOCH_COMPLETED)
        if engine.should_terminate:
            break
    assert engine.epoch == 12
    assert handler.state_dict() == {
        "waiting_counter": 5,
        "best_epoch": 7,
        "best_score": 3.5,
    }


def test_early_stopping_simulation_delta():
    engine = create_engine()
    engine.add_history(MinScalarHistory(f"{ct.EVAL}/loss"))
    handler = EarlyStopping(delta=0.15)
    handler.attach(engine)
    # Simulate the behavior of the engine.
    for loss in [3, 2, 1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]:
        engine.increment_epoch()
        engine.log_metric(f"{ct.EVAL}/loss", loss)
        engine.fire_event(EngineEvents.EPOCH_COMPLETED)
        if engine.should_terminate:
            break
    assert engine.epoch == 7
    assert handler.state_dict() == {
        "waiting_counter": 5,
        "best_epoch": 2,
        "best_score": 0.5,
    }


def test_early_stopping_simulation_delta_and_cumulative_delta():
    engine = create_engine()
    engine.add_history(MinScalarHistory(f"{ct.EVAL}/loss"))
    handler = EarlyStopping(delta=0.15, cumulative_delta=True)
    handler.attach(engine)
    # Simulate the behavior of the engine.
    for loss in [3, 2, 1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]:
        engine.increment_epoch()
        engine.log_metric(f"{ct.EVAL}/loss", loss)
        engine.fire_event(EngineEvents.EPOCH_COMPLETED)
        if engine.should_terminate:
            break
    assert engine.epoch == 11
    assert handler.state_dict() == {
        "waiting_counter": 5,
        "best_epoch": 6,
        "best_score": 0.6,
    }


def test_early_stopping_simulation_resume():
    engine = create_engine(state=VanillaEngineState(epoch=10))
    engine.add_history(MinScalarHistory(f"{ct.EVAL}/loss"))
    handler = EarlyStopping()
    handler.attach(engine)
    handler.load_state_dict(
        {
            "waiting_counter": 0,
            "best_epoch": 10,
            "best_score": 0.2,
        }
    )
    # Simulate the behavior of the engine.
    for loss in [3, 2, 1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]:
        engine.increment_epoch()
        engine.log_metric(f"{ct.EVAL}/loss", loss)
        engine.fire_event(EngineEvents.EPOCH_COMPLETED)
        if engine.should_terminate:
            break
    assert engine.epoch == 15
    assert handler.state_dict() == {
        "waiting_counter": 5,
        "best_epoch": 10,
        "best_score": 0.2,
    }


def test_early_stopping_simulation_resume_cumulative_delta():
    engine = create_engine(state=VanillaEngineState(epoch=10))
    engine.add_history(MinScalarHistory(f"{ct.EVAL}/loss"))
    handler = EarlyStopping(delta=0.5, cumulative_delta=True)
    handler.attach(engine)
    handler.load_state_dict(
        {
            "waiting_counter": 2,
            "best_epoch": 9,
            "best_score": 1.1,
        }
    )
    # Simulate the behavior of the engine.
    for loss in [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]:
        engine.increment_epoch()
        engine.log_metric(f"{ct.EVAL}/loss", loss)
        engine.fire_event(EngineEvents.EPOCH_COMPLETED)
        if engine.should_terminate:
            break
    assert engine.epoch == 13
    assert handler.state_dict() == {
        "waiting_counter": 5,
        "best_epoch": 9,
        "best_score": 1.1,
    }
