import math
from unittest.mock import Mock

import torch
from coola import objects_are_allclose
from pytest import fixture, mark, raises

from gravitorch import constants as ct
from gravitorch.engines import BaseEngine, EngineEvents
from gravitorch.models.metrics import (
    EmptyMetricError,
    RootMeanSquaredError,
    SquaredError,
)
from gravitorch.models.metrics.state import (
    BaseState,
    ErrorState,
    ExtendedErrorState,
    MeanErrorState,
)
from gravitorch.testing import create_dummy_engine
from gravitorch.utils import get_available_devices
from gravitorch.utils.events import GEventHandler
from gravitorch.utils.history import MinScalarHistory

MODES = (ct.TRAIN, ct.EVAL)
NAMES = ("sq_err", "squared_error")
SIZES = (1, 2)
DTYPES = (torch.long, torch.float)


@fixture
def engine() -> BaseEngine:
    return create_dummy_engine()


##################################
#     Tests for SquaredError     #
##################################


@mark.parametrize("mode", MODES)
def test_squared_error_str(mode: str) -> None:
    assert str(SquaredError(mode)).startswith("SquaredError(")


def test_squared_error_init_state_default() -> None:
    assert isinstance(SquaredError(ct.EVAL)._state, ErrorState)


def test_squared_error_init_state_mean() -> None:
    assert isinstance(SquaredError(ct.EVAL, state=MeanErrorState())._state, MeanErrorState)


@mark.parametrize("name", NAMES)
def test_squared_error_attach_train(name: str, engine: BaseEngine) -> None:
    metric = SquaredError(ct.TRAIN, name=name)
    metric.attach(engine)
    assert isinstance(engine.get_history(f"{ct.TRAIN}/{name}_mean"), MinScalarHistory)
    assert isinstance(engine.get_history(f"{ct.TRAIN}/{name}_max"), MinScalarHistory)
    assert isinstance(engine.get_history(f"{ct.TRAIN}/{name}_min"), MinScalarHistory)
    assert isinstance(engine.get_history(f"{ct.TRAIN}/{name}_sum"), MinScalarHistory)
    assert engine.has_event_handler(GEventHandler(metric.reset), EngineEvents.TRAIN_EPOCH_STARTED)
    assert engine.has_event_handler(
        GEventHandler(metric.value, handler_kwargs={"engine": engine}),
        EngineEvents.TRAIN_EPOCH_COMPLETED,
    )


@mark.parametrize("name", NAMES)
def test_squared_error_attach_eval(name: str, engine: BaseEngine) -> None:
    metric = SquaredError(ct.EVAL, name=name)
    metric.attach(engine)
    assert isinstance(engine.get_history(f"{ct.EVAL}/{name}_mean"), MinScalarHistory)
    assert isinstance(engine.get_history(f"{ct.EVAL}/{name}_max"), MinScalarHistory)
    assert isinstance(engine.get_history(f"{ct.EVAL}/{name}_min"), MinScalarHistory)
    assert isinstance(engine.get_history(f"{ct.EVAL}/{name}_sum"), MinScalarHistory)
    assert engine.has_event_handler(GEventHandler(metric.reset), EngineEvents.EVAL_EPOCH_STARTED)
    assert engine.has_event_handler(
        GEventHandler(metric.value, handler_kwargs={"engine": engine}),
        EngineEvents.EVAL_EPOCH_COMPLETED,
    )


def test_squared_error_attach_state_mean(engine: BaseEngine) -> None:
    metric = SquaredError(ct.EVAL, state=MeanErrorState())
    metric.attach(engine)
    assert isinstance(engine.get_history(f"{ct.EVAL}/sq_err_mean"), MinScalarHistory)
    assert engine.has_event_handler(GEventHandler(metric.reset), EngineEvents.EVAL_EPOCH_STARTED)
    assert engine.has_event_handler(
        GEventHandler(metric.value, handler_kwargs={"engine": engine}),
        EngineEvents.EVAL_EPOCH_COMPLETED,
    )


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("feature_size", SIZES)
def test_squared_error_forward_correct(
    device: str, mode: str, batch_size: int, feature_size: int
) -> None:
    device = torch.device(device)
    metric = SquaredError(mode).to(device=device)
    metric(
        torch.ones(batch_size, feature_size, device=device),
        torch.ones(batch_size, feature_size, device=device),
    )
    assert metric.value() == {
        f"{mode}/sq_err_mean": 0.0,
        f"{mode}/sq_err_max": 0.0,
        f"{mode}/sq_err_min": 0.0,
        f"{mode}/sq_err_sum": 0.0,
        f"{mode}/sq_err_num_predictions": batch_size * feature_size,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("feature_size", SIZES)
def test_squared_error_forward_incorrect(
    device: str, mode: str, batch_size: int, feature_size: int
) -> None:
    device = torch.device(device)
    metric = SquaredError(mode).to(device=device)
    metric(
        -torch.ones(batch_size, feature_size, device=device),
        torch.ones(batch_size, feature_size, device=device),
    )
    assert metric.value() == {
        f"{mode}/sq_err_mean": 4.0,
        f"{mode}/sq_err_max": 4.0,
        f"{mode}/sq_err_min": 4.0,
        f"{mode}/sq_err_sum": 4.0 * batch_size * feature_size,
        f"{mode}/sq_err_num_predictions": batch_size * feature_size,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_squared_error_forward_partially_correct(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = SquaredError(mode).to(device=device)
    metric(torch.eye(2, device=device), -torch.eye(2, device=device))
    assert metric.value() == {
        f"{mode}/sq_err_mean": 2.0,
        f"{mode}/sq_err_max": 4.0,
        f"{mode}/sq_err_min": 0.0,
        f"{mode}/sq_err_sum": 8.0,
        f"{mode}/sq_err_num_predictions": 4,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_squared_error_forward_1d(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = SquaredError(mode).to(device=device)
    metric(torch.ones(2, device=device), torch.ones(2, device=device))
    assert metric.value() == {
        f"{mode}/sq_err_mean": 0.0,
        f"{mode}/sq_err_max": 0.0,
        f"{mode}/sq_err_min": 0.0,
        f"{mode}/sq_err_sum": 0.0,
        f"{mode}/sq_err_num_predictions": 2,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_squared_error_forward_2d(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = SquaredError(mode).to(device=device)
    metric(torch.ones(2, 3, device=device), torch.ones(2, 3, device=device))
    assert metric.value() == {
        f"{mode}/sq_err_mean": 0.0,
        f"{mode}/sq_err_max": 0.0,
        f"{mode}/sq_err_min": 0.0,
        f"{mode}/sq_err_sum": 0.0,
        f"{mode}/sq_err_num_predictions": 6,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_squared_error_forward_3d(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = SquaredError(mode).to(device=device)
    metric(torch.ones(2, 3, 4, device=device), torch.ones(2, 3, 4, device=device))
    assert metric.value() == {
        f"{mode}/sq_err_mean": 0.0,
        f"{mode}/sq_err_max": 0.0,
        f"{mode}/sq_err_min": 0.0,
        f"{mode}/sq_err_sum": 0.0,
        f"{mode}/sq_err_num_predictions": 24,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
@mark.parametrize("dtype_prediction", DTYPES)
@mark.parametrize("dtype_target", DTYPES)
def test_squared_error_forward_dtypes(
    device: str,
    mode: str,
    dtype_prediction: torch.dtype,
    dtype_target: torch.dtype,
) -> None:
    device = torch.device(device)
    metric = SquaredError(mode).to(device=device)
    metric(
        torch.ones(2, 2, device=device, dtype=dtype_prediction),
        torch.ones(2, 2, device=device, dtype=dtype_target),
    )
    assert metric.value() == {
        f"{mode}/sq_err_mean": 0.0,
        f"{mode}/sq_err_max": 0.0,
        f"{mode}/sq_err_min": 0.0,
        f"{mode}/sq_err_sum": 0.0,
        f"{mode}/sq_err_num_predictions": 4,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_squared_error_forward_state(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = SquaredError(mode, state=ExtendedErrorState()).to(device=device)
    metric(torch.ones(2, 2, device=device), torch.ones(2, 2, device=device))
    assert metric.value() == {
        f"{mode}/sq_err_mean": 0.0,
        f"{mode}/sq_err_max": 0.0,
        f"{mode}/sq_err_min": 0.0,
        f"{mode}/sq_err_sum": 0.0,
        f"{mode}/sq_err_std": 0.0,
        f"{mode}/sq_err_median": 0.0,
        f"{mode}/sq_err_num_predictions": 4,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_squared_error_forward_multiple_batches(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = SquaredError(mode).to(device=device)
    metric(torch.ones(2, 2, device=device), torch.ones(2, 2, device=device))
    metric(torch.eye(2, device=device), -torch.eye(2, device=device))
    assert metric.value() == {
        f"{mode}/sq_err_mean": 1.0,
        f"{mode}/sq_err_max": 4.0,
        f"{mode}/sq_err_min": 0.0,
        f"{mode}/sq_err_sum": 8.0,
        f"{mode}/sq_err_num_predictions": 8,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_squared_error_forward_multiple_batches_with_reset(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = SquaredError(mode).to(device=device)
    metric(torch.ones(2, 2, device=device), torch.ones(2, 2, device=device))
    metric.reset()
    metric(torch.eye(2, device=device), -torch.eye(2, device=device))
    assert metric.value() == {
        f"{mode}/sq_err_mean": 2.0,
        f"{mode}/sq_err_max": 4.0,
        f"{mode}/sq_err_min": 0.0,
        f"{mode}/sq_err_sum": 8.0,
        f"{mode}/sq_err_num_predictions": 4,
    }


@mark.parametrize("mode", MODES)
def test_squared_error_value_empty(mode: str) -> None:
    with raises(EmptyMetricError, match="ErrorState is empty"):
        SquaredError(mode).value()


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_squared_error_value_log_engine(device: str, mode: str, engine: BaseEngine) -> None:
    device = torch.device(device)
    metric = SquaredError(mode).to(device=device)
    metric(torch.eye(2, device=device), -torch.eye(2, device=device))
    metric.value(engine)
    assert engine.get_history(f"{mode}/sq_err_mean").get_last_value() == 2.0
    assert engine.get_history(f"{mode}/sq_err_max").get_last_value() == 4.0
    assert engine.get_history(f"{mode}/sq_err_min").get_last_value() == 0.0
    assert engine.get_history(f"{mode}/sq_err_sum").get_last_value() == 8.0
    assert engine.get_history(f"{mode}/sq_err_num_predictions").get_last_value() == 4


@mark.parametrize("device", get_available_devices())
def test_squared_error_events_train(device: str, engine: BaseEngine) -> None:
    device = torch.device(device)
    metric = SquaredError(ct.TRAIN).to(device=device)
    metric.attach(engine)
    metric(torch.ones(2, 2, device=device), torch.ones(2, 2, device=device))
    engine.fire_event(EngineEvents.TRAIN_EPOCH_STARTED)
    metric(torch.eye(2, device=device), -torch.eye(2, device=device))
    engine.fire_event(EngineEvents.TRAIN_EPOCH_COMPLETED)
    assert engine.get_history(f"{ct.TRAIN}/sq_err_mean").get_last_value() == 2.0
    assert engine.get_history(f"{ct.TRAIN}/sq_err_max").get_last_value() == 4.0
    assert engine.get_history(f"{ct.TRAIN}/sq_err_min").get_last_value() == 0.0
    assert engine.get_history(f"{ct.TRAIN}/sq_err_sum").get_last_value() == 8.0
    assert engine.get_history(f"{ct.TRAIN}/sq_err_num_predictions").get_last_value() == 4


@mark.parametrize("device", get_available_devices())
def test_squared_error_events_eval(device: str, engine: BaseEngine) -> None:
    device = torch.device(device)
    metric = SquaredError(ct.EVAL).to(device=device)
    metric.attach(engine)
    metric(torch.ones(2, 2, device=device), torch.ones(2, 2, device=device))
    engine.fire_event(EngineEvents.EVAL_EPOCH_STARTED)
    metric(torch.eye(2, device=device), -torch.eye(2, device=device))
    engine.fire_event(EngineEvents.EVAL_EPOCH_COMPLETED)
    assert engine.get_history(f"{ct.EVAL}/sq_err_mean").get_last_value() == 2.0
    assert engine.get_history(f"{ct.EVAL}/sq_err_max").get_last_value() == 4.0
    assert engine.get_history(f"{ct.EVAL}/sq_err_min").get_last_value() == 0.0
    assert engine.get_history(f"{ct.EVAL}/sq_err_sum").get_last_value() == 8.0
    assert engine.get_history(f"{ct.EVAL}/sq_err_num_predictions").get_last_value() == 4


def test_squared_error_reset() -> None:
    state = Mock(spec=BaseState)
    metric = SquaredError(ct.EVAL, state=state)
    metric.reset()
    state.reset.assert_called_once_with()


##########################################
#     Tests for RootMeanSquaredError     #
##########################################


@mark.parametrize("mode", MODES)
def test_root_mean_squared_error_str(mode: str) -> None:
    assert str(RootMeanSquaredError(mode)).startswith("RootMeanSquaredError(")


@mark.parametrize("name", NAMES)
def test_root_mean_squared_error_attach_train(name: str, engine: BaseEngine) -> None:
    metric = RootMeanSquaredError(ct.TRAIN, name=name)
    metric.attach(engine)
    assert isinstance(engine.get_history(f"{ct.TRAIN}/{name}_root_mean"), MinScalarHistory)
    assert engine.has_event_handler(GEventHandler(metric.reset), EngineEvents.TRAIN_EPOCH_STARTED)
    assert engine.has_event_handler(
        GEventHandler(metric.value, handler_kwargs={"engine": engine}),
        EngineEvents.TRAIN_EPOCH_COMPLETED,
    )


@mark.parametrize("name", NAMES)
def test_root_mean_squared_error_attach_eval(name: str, engine: BaseEngine) -> None:
    metric = RootMeanSquaredError(ct.EVAL, name=name)
    metric.attach(engine)
    assert isinstance(engine.get_history(f"{ct.EVAL}/{name}_root_mean"), MinScalarHistory)
    assert engine.has_event_handler(GEventHandler(metric.reset), EngineEvents.EVAL_EPOCH_STARTED)
    assert engine.has_event_handler(
        GEventHandler(metric.value, handler_kwargs={"engine": engine}),
        EngineEvents.EVAL_EPOCH_COMPLETED,
    )


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("feature_size", SIZES)
def test_root_mean_squared_error_forward_correct(
    device: str, mode: str, batch_size: int, feature_size: int
) -> None:
    device = torch.device(device)
    metric = RootMeanSquaredError(mode).to(device=device)
    metric(
        torch.ones(batch_size, feature_size, device=device),
        torch.ones(batch_size, feature_size, device=device),
    )
    assert metric.value() == {
        f"{mode}/rmse_root_mean": 0.0,
        f"{mode}/rmse_num_predictions": batch_size * feature_size,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("feature_size", SIZES)
def test_root_mean_squared_error_forward_incorrect(
    device: str, mode: str, batch_size: int, feature_size: int
) -> None:
    device = torch.device(device)
    metric = RootMeanSquaredError(mode).to(device=device)
    metric(
        torch.ones(batch_size, feature_size, device=device),
        -torch.ones(batch_size, feature_size, device=device),
    )
    assert metric.value() == {
        f"{mode}/rmse_root_mean": 2.0,
        f"{mode}/rmse_num_predictions": batch_size * feature_size,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_root_mean_squared_error_forward_partially_correct(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = RootMeanSquaredError(mode).to(device=device)
    metric(torch.eye(2, device=device), -torch.eye(2, device=device))
    assert objects_are_allclose(
        metric.value(),
        {
            f"{mode}/rmse_root_mean": math.sqrt(2.0),
            f"{mode}/rmse_num_predictions": 4,
        },
    )


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_root_mean_squared_error_forward_1d(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = RootMeanSquaredError(mode).to(device=device)
    metric(torch.ones(2, device=device), torch.ones(2, device=device))
    assert metric.value() == {
        f"{mode}/rmse_root_mean": 0.0,
        f"{mode}/rmse_num_predictions": 2,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_root_mean_squared_error_forward_2d(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = RootMeanSquaredError(mode).to(device=device)
    metric(torch.ones(2, 3, device=device), torch.ones(2, 3, device=device))
    assert metric.value() == {
        f"{mode}/rmse_root_mean": 0.0,
        f"{mode}/rmse_num_predictions": 6,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_root_mean_squared_error_forward_3d(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = RootMeanSquaredError(mode).to(device=device)
    metric(torch.ones(2, 3, 4, device=device), torch.ones(2, 3, 4, device=device))
    assert metric.value() == {
        f"{mode}/rmse_root_mean": 0.0,
        f"{mode}/rmse_num_predictions": 24,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
@mark.parametrize("dtype_prediction", DTYPES)
@mark.parametrize("dtype_target", DTYPES)
def test_root_mean_squared_error_forward_dtypes(
    device: str,
    mode: str,
    dtype_prediction: torch.dtype,
    dtype_target: torch.dtype,
) -> None:
    device = torch.device(device)
    metric = RootMeanSquaredError(mode).to(device=device)
    metric(
        torch.ones(2, 2, device=device, dtype=dtype_prediction),
        torch.ones(2, 2, device=device, dtype=dtype_target),
    )
    assert metric.value() == {
        f"{mode}/rmse_root_mean": 0.0,
        f"{mode}/rmse_num_predictions": 4,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_root_mean_squared_error_forward_multiple_batches(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = RootMeanSquaredError(mode).to(device=device)
    metric(torch.ones(2, 2, device=device), torch.ones(2, 2, device=device))
    metric(torch.eye(2, device=device), -torch.eye(2, device=device))
    assert metric.value() == {
        f"{mode}/rmse_root_mean": 1.0,
        f"{mode}/rmse_num_predictions": 8,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_root_mean_squared_error_forward_multiple_batches_with_reset(
    device: str, mode: str
) -> None:
    device = torch.device(device)
    metric = RootMeanSquaredError(mode).to(device=device)
    metric(torch.ones(2, 2, device=device), torch.ones(2, 2, device=device))
    metric.reset()
    metric(torch.eye(2, device=device), -torch.eye(2, device=device))
    assert objects_are_allclose(
        metric.value(),
        {
            f"{mode}/rmse_root_mean": math.sqrt(2.0),
            f"{mode}/rmse_num_predictions": 4,
        },
    )


@mark.parametrize("mode", MODES)
def test_root_mean_squared_error_value_empty(mode: str) -> None:
    with raises(EmptyMetricError, match="ErrorState is empty"):
        RootMeanSquaredError(mode).value()


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_root_mean_squared_error_value_log_engine(
    device: str, mode: str, engine: BaseEngine
) -> None:
    device = torch.device(device)
    metric = RootMeanSquaredError(mode).to(device=device)
    metric(torch.eye(2, device=device), torch.eye(2, device=device))
    metric.value(engine)
    assert engine.get_history(f"{mode}/rmse_root_mean").get_last_value() == 0.0
    assert engine.get_history(f"{mode}/rmse_num_predictions").get_last_value() == 4


@mark.parametrize("device", get_available_devices())
def test_root_mean_squared_error_events_train(device: str, engine: BaseEngine) -> None:
    device = torch.device(device)
    metric = RootMeanSquaredError(ct.TRAIN).to(device=device)
    metric.attach(engine)
    metric(torch.ones(2, 2, device=device), torch.ones(2, 2, device=device))
    engine.fire_event(EngineEvents.TRAIN_EPOCH_STARTED)
    metric(torch.eye(2, device=device), torch.eye(2, device=device))
    engine.fire_event(EngineEvents.TRAIN_EPOCH_COMPLETED)
    assert engine.get_history(f"{ct.TRAIN}/rmse_root_mean").get_last_value() == 0.0
    assert engine.get_history(f"{ct.TRAIN}/rmse_num_predictions").get_last_value() == 4


@mark.parametrize("device", get_available_devices())
def test_root_mean_squared_error_events_eval(device: str, engine: BaseEngine) -> None:
    device = torch.device(device)
    metric = RootMeanSquaredError(ct.EVAL).to(device=device)
    metric.attach(engine)
    metric(torch.ones(2, 2, device=device), torch.ones(2, 2, device=device))
    engine.fire_event(EngineEvents.EVAL_EPOCH_STARTED)
    metric(torch.eye(2, device=device), torch.eye(2, device=device))
    engine.fire_event(EngineEvents.EVAL_EPOCH_COMPLETED)
    assert engine.get_history(f"{ct.EVAL}/rmse_root_mean").get_last_value() == 0.0
    assert engine.get_history(f"{ct.EVAL}/rmse_num_predictions").get_last_value() == 4


def test_root_mean_squared_error_reset() -> None:
    state = Mock(spec=BaseState)
    metric = RootMeanSquaredError(ct.EVAL)
    metric._state = state
    metric.reset()
    state.reset.assert_called_once_with()
