from unittest.mock import Mock

import torch
from coola import objects_are_allclose
from pytest import fixture, mark, raises

from gravitorch import constants as ct
from gravitorch.engines import BaseEngine, EngineEvents
from gravitorch.events import VanillaEventHandler
from gravitorch.models.metrics import (
    EmptyMetricError,
    SquaredAsinhError,
    SquaredLogError,
    SquaredSymlogError,
)
from gravitorch.models.metrics.state import (
    BaseState,
    ErrorState,
    ExtendedErrorState,
    MeanErrorState,
)
from gravitorch.testing import create_dummy_engine
from gravitorch.utils import get_available_devices
from gravitorch.utils.history import MinScalarHistory

MODES = (ct.TRAIN, ct.EVAL)
NAMES = ("sq_log_err", "squared_log_error")
SIZES = (1, 2)
DTYPES = (torch.long, torch.float)


@fixture
def engine() -> BaseEngine:
    return create_dummy_engine()


#####################################
#     Tests for SquaredLogError     #
#####################################


@mark.parametrize("mode", MODES)
def test_squared_log_error_str(mode: str) -> None:
    assert str(SquaredLogError(mode)).startswith("SquaredLogError(")


def test_squared_log_error_init_state_default() -> None:
    assert isinstance(SquaredLogError(ct.EVAL)._state, ErrorState)


def test_squared_log_error_init_state_mean() -> None:
    assert isinstance(SquaredLogError(ct.EVAL, state=MeanErrorState())._state, MeanErrorState)


@mark.parametrize("name", NAMES)
def test_squared_log_error_attach_train(name: str, engine: BaseEngine) -> None:
    metric = SquaredLogError(ct.TRAIN, name=name)
    metric.attach(engine)
    assert isinstance(engine.get_history(f"{ct.TRAIN}/{name}_mean"), MinScalarHistory)
    assert isinstance(engine.get_history(f"{ct.TRAIN}/{name}_max"), MinScalarHistory)
    assert isinstance(engine.get_history(f"{ct.TRAIN}/{name}_min"), MinScalarHistory)
    assert isinstance(engine.get_history(f"{ct.TRAIN}/{name}_sum"), MinScalarHistory)
    assert engine.has_event_handler(
        VanillaEventHandler(metric.reset), EngineEvents.TRAIN_EPOCH_STARTED
    )
    assert engine.has_event_handler(
        VanillaEventHandler(metric.value, handler_kwargs={"engine": engine}),
        EngineEvents.TRAIN_EPOCH_COMPLETED,
    )


@mark.parametrize("name", NAMES)
def test_squared_log_error_attach_eval(name: str, engine: BaseEngine) -> None:
    metric = SquaredLogError(ct.EVAL, name=name)
    metric.attach(engine)
    assert isinstance(engine.get_history(f"{ct.EVAL}/{name}_mean"), MinScalarHistory)
    assert isinstance(engine.get_history(f"{ct.EVAL}/{name}_max"), MinScalarHistory)
    assert isinstance(engine.get_history(f"{ct.EVAL}/{name}_min"), MinScalarHistory)
    assert isinstance(engine.get_history(f"{ct.EVAL}/{name}_sum"), MinScalarHistory)
    assert engine.has_event_handler(
        VanillaEventHandler(metric.reset), EngineEvents.EVAL_EPOCH_STARTED
    )
    assert engine.has_event_handler(
        VanillaEventHandler(metric.value, handler_kwargs={"engine": engine}),
        EngineEvents.EVAL_EPOCH_COMPLETED,
    )


def test_squared_log_error_attach_state_mean(engine: BaseEngine) -> None:
    metric = SquaredLogError(ct.EVAL, state=MeanErrorState())
    metric.attach(engine)
    assert isinstance(engine.get_history(f"{ct.EVAL}/sq_log_err_mean"), MinScalarHistory)
    assert engine.has_event_handler(
        VanillaEventHandler(metric.reset), EngineEvents.EVAL_EPOCH_STARTED
    )
    assert engine.has_event_handler(
        VanillaEventHandler(metric.value, handler_kwargs={"engine": engine}),
        EngineEvents.EVAL_EPOCH_COMPLETED,
    )


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("feature_size", SIZES)
def test_squared_log_error_forward_correct(
    device: str, mode: str, batch_size: int, feature_size: int
) -> None:
    device = torch.device(device)
    metric = SquaredLogError(mode).to(device=device)
    metric(
        torch.ones(batch_size, feature_size, device=device),
        torch.ones(batch_size, feature_size, device=device),
    )
    assert metric.value() == {
        f"{mode}/sq_log_err_mean": 0.0,
        f"{mode}/sq_log_err_max": 0.0,
        f"{mode}/sq_log_err_min": 0.0,
        f"{mode}/sq_log_err_sum": 0.0,
        f"{mode}/sq_log_err_num_predictions": batch_size * feature_size,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("feature_size", SIZES)
def test_squared_log_error_forward_incorrect(
    device: str, mode: str, batch_size: int, feature_size: int
) -> None:
    device = torch.device(device)
    metric = SquaredLogError(mode).to(device=device)
    metric(
        2 * torch.ones(batch_size, feature_size, device=device),
        torch.ones(batch_size, feature_size, device=device),
    )
    assert objects_are_allclose(
        metric.value(),
        {
            f"{mode}/sq_log_err_mean": 0.16440195389316548,
            f"{mode}/sq_log_err_max": 0.16440195389316548,
            f"{mode}/sq_log_err_min": 0.16440195389316548,
            f"{mode}/sq_log_err_sum": 0.16440195389316548 * batch_size * feature_size,
            f"{mode}/sq_log_err_num_predictions": batch_size * feature_size,
        },
    )


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_squared_log_error_forward_partially_correct(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = SquaredLogError(mode).to(device=device)
    metric(torch.ones(2, 2, device=device), torch.eye(2, device=device))
    assert objects_are_allclose(
        metric.value(),
        {
            f"{mode}/sq_log_err_mean": 0.2402265069591007,
            f"{mode}/sq_log_err_max": 0.4804530139182014,
            f"{mode}/sq_log_err_min": 0.0,
            f"{mode}/sq_log_err_sum": 0.9609060278364028,
            f"{mode}/sq_log_err_num_predictions": 4,
        },
    )


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_squared_log_error_forward_1d(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = SquaredLogError(mode).to(device=device)
    metric(torch.ones(2, device=device), torch.ones(2, device=device))
    assert metric.value() == {
        f"{mode}/sq_log_err_mean": 0.0,
        f"{mode}/sq_log_err_max": 0.0,
        f"{mode}/sq_log_err_min": 0.0,
        f"{mode}/sq_log_err_sum": 0.0,
        f"{mode}/sq_log_err_num_predictions": 2,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_squared_log_error_forward_2d(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = SquaredLogError(mode).to(device=device)
    metric(torch.ones(2, 3, device=device), torch.ones(2, 3, device=device))
    assert metric.value() == {
        f"{mode}/sq_log_err_mean": 0.0,
        f"{mode}/sq_log_err_max": 0.0,
        f"{mode}/sq_log_err_min": 0.0,
        f"{mode}/sq_log_err_sum": 0.0,
        f"{mode}/sq_log_err_num_predictions": 6,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_squared_log_error_forward_3d(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = SquaredLogError(mode).to(device=device)
    metric(torch.ones(2, 3, 4, device=device), torch.ones(2, 3, 4, device=device))
    assert metric.value() == {
        f"{mode}/sq_log_err_mean": 0.0,
        f"{mode}/sq_log_err_max": 0.0,
        f"{mode}/sq_log_err_min": 0.0,
        f"{mode}/sq_log_err_sum": 0.0,
        f"{mode}/sq_log_err_num_predictions": 24,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
@mark.parametrize("dtype_prediction", DTYPES)
@mark.parametrize("dtype_target", DTYPES)
def test_squared_log_error_forward_dtypes(
    device: str,
    mode: str,
    dtype_prediction: torch.dtype,
    dtype_target: torch.dtype,
) -> None:
    device = torch.device(device)
    metric = SquaredLogError(mode).to(device=device)
    metric(
        torch.ones(2, 2, device=device, dtype=dtype_prediction),
        torch.ones(2, 2, device=device, dtype=dtype_target),
    )
    assert metric.value() == {
        f"{mode}/sq_log_err_mean": 0.0,
        f"{mode}/sq_log_err_max": 0.0,
        f"{mode}/sq_log_err_min": 0.0,
        f"{mode}/sq_log_err_sum": 0.0,
        f"{mode}/sq_log_err_num_predictions": 4,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_squared_log_error_forward_state(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = SquaredLogError(mode, state=ExtendedErrorState()).to(device=device)
    metric(torch.ones(2, 2, device=device), torch.ones(2, 2, device=device))
    assert metric.value() == {
        f"{mode}/sq_log_err_mean": 0.0,
        f"{mode}/sq_log_err_max": 0.0,
        f"{mode}/sq_log_err_min": 0.0,
        f"{mode}/sq_log_err_sum": 0.0,
        f"{mode}/sq_log_err_std": 0.0,
        f"{mode}/sq_log_err_median": 0.0,
        f"{mode}/sq_log_err_num_predictions": 4,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_squared_log_error_forward_multiple_batches(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = SquaredLogError(mode).to(device=device)
    metric(torch.ones(2, 2, device=device), torch.ones(2, 2, device=device))
    metric(torch.ones(2, 2, device=device), torch.eye(2, device=device))
    assert objects_are_allclose(
        metric.value(),
        {
            f"{mode}/sq_log_err_mean": 0.12011325347955035,
            f"{mode}/sq_log_err_max": 0.4804530139182014,
            f"{mode}/sq_log_err_min": 0.0,
            f"{mode}/sq_log_err_sum": 0.9609060278364028,
            f"{mode}/sq_log_err_num_predictions": 8,
        },
    )


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_squared_log_error_forward_multiple_batches_with_reset(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = SquaredLogError(mode).to(device=device)
    metric(torch.ones(2, 2, device=device), torch.ones(2, 2, device=device))
    metric.reset()
    metric(torch.ones(2, 2, device=device), torch.eye(2, device=device))
    assert objects_are_allclose(
        metric.value(),
        {
            f"{mode}/sq_log_err_mean": 0.2402265069591007,
            f"{mode}/sq_log_err_max": 0.4804530139182014,
            f"{mode}/sq_log_err_min": 0.0,
            f"{mode}/sq_log_err_sum": 0.9609060278364028,
            f"{mode}/sq_log_err_num_predictions": 4,
        },
    )


@mark.parametrize("mode", MODES)
def test_squared_log_error_value_empty(mode: str) -> None:
    with raises(EmptyMetricError):
        SquaredLogError(mode).value()


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_squared_log_error_value_log_engine(device: str, mode: str, engine: BaseEngine) -> None:
    device = torch.device(device)
    metric = SquaredLogError(mode).to(device=device)
    metric(torch.ones(2, 2, device=device), torch.ones(2, 2, device=device))
    metric.value(engine)
    assert engine.get_history(f"{mode}/sq_log_err_mean").get_last_value() == 0.0
    assert engine.get_history(f"{mode}/sq_log_err_max").get_last_value() == 0.0
    assert engine.get_history(f"{mode}/sq_log_err_min").get_last_value() == 0.0
    assert engine.get_history(f"{mode}/sq_log_err_sum").get_last_value() == 0.0
    assert engine.get_history(f"{mode}/sq_log_err_num_predictions").get_last_value() == 4


@mark.parametrize("device", get_available_devices())
def test_squared_log_error_events_train(device: str, engine: BaseEngine) -> None:
    device = torch.device(device)
    metric = SquaredLogError(ct.TRAIN).to(device=device)
    metric.attach(engine)
    metric(torch.ones(2, 2, device=device), 2 * torch.ones(2, 2, device=device))
    engine.fire_event(EngineEvents.TRAIN_EPOCH_STARTED)
    metric(torch.eye(2, device=device), torch.eye(2, device=device))
    engine.fire_event(EngineEvents.TRAIN_EPOCH_COMPLETED)
    assert engine.get_history(f"{ct.TRAIN}/sq_log_err_mean").get_last_value() == 0.0
    assert engine.get_history(f"{ct.TRAIN}/sq_log_err_max").get_last_value() == 0.0
    assert engine.get_history(f"{ct.TRAIN}/sq_log_err_min").get_last_value() == 0.0
    assert engine.get_history(f"{ct.TRAIN}/sq_log_err_sum").get_last_value() == 0.0
    assert engine.get_history(f"{ct.TRAIN}/sq_log_err_num_predictions").get_last_value() == 4


@mark.parametrize("device", get_available_devices())
def test_squared_log_error_events_eval(device: str, engine: BaseEngine) -> None:
    device = torch.device(device)
    metric = SquaredLogError(ct.EVAL).to(device=device)
    metric.attach(engine)
    metric(torch.ones(2, 2, device=device), 2 * torch.ones(2, 2, device=device))
    engine.fire_event(EngineEvents.EVAL_EPOCH_STARTED)
    metric(torch.eye(2, device=device), torch.eye(2, device=device))
    engine.fire_event(EngineEvents.EVAL_EPOCH_COMPLETED)
    assert engine.get_history(f"{ct.EVAL}/sq_log_err_mean").get_last_value() == 0.0
    assert engine.get_history(f"{ct.EVAL}/sq_log_err_max").get_last_value() == 0.0
    assert engine.get_history(f"{ct.EVAL}/sq_log_err_min").get_last_value() == 0.0
    assert engine.get_history(f"{ct.EVAL}/sq_log_err_sum").get_last_value() == 0.0
    assert engine.get_history(f"{ct.EVAL}/sq_log_err_num_predictions").get_last_value() == 4


def test_squared_log_error_reset() -> None:
    state = Mock(spec=BaseState)
    metric = SquaredLogError(ct.EVAL, state=state)
    metric.reset()
    state.reset.assert_called_once_with()


########################################
#     Tests for SquaredSymlogError     #
########################################


@mark.parametrize("mode", MODES)
def test_squared_symlog_error_str(mode: str) -> None:
    assert str(SquaredSymlogError(mode)).startswith("SquaredSymlogError(")


def test_squared_symlog_error_init_state_default() -> None:
    assert isinstance(SquaredSymlogError(ct.EVAL)._state, ErrorState)


def test_squared_symlog_error_init_state_mean() -> None:
    assert isinstance(SquaredSymlogError(ct.EVAL, state=MeanErrorState())._state, MeanErrorState)


@mark.parametrize("name", NAMES)
def test_squared_symlog_error_attach_train(name: str, engine: BaseEngine) -> None:
    metric = SquaredSymlogError(ct.TRAIN, name=name)
    metric.attach(engine)
    assert isinstance(engine.get_history(f"{ct.TRAIN}/{name}_mean"), MinScalarHistory)
    assert isinstance(engine.get_history(f"{ct.TRAIN}/{name}_max"), MinScalarHistory)
    assert isinstance(engine.get_history(f"{ct.TRAIN}/{name}_min"), MinScalarHistory)
    assert isinstance(engine.get_history(f"{ct.TRAIN}/{name}_sum"), MinScalarHistory)
    assert engine.has_event_handler(
        VanillaEventHandler(metric.reset), EngineEvents.TRAIN_EPOCH_STARTED
    )
    assert engine.has_event_handler(
        VanillaEventHandler(metric.value, handler_kwargs={"engine": engine}),
        EngineEvents.TRAIN_EPOCH_COMPLETED,
    )


@mark.parametrize("name", NAMES)
def test_squared_symlog_error_attach_eval(name: str, engine: BaseEngine) -> None:
    metric = SquaredSymlogError(ct.EVAL, name=name)
    metric.attach(engine)
    assert isinstance(engine.get_history(f"{ct.EVAL}/{name}_mean"), MinScalarHistory)
    assert isinstance(engine.get_history(f"{ct.EVAL}/{name}_max"), MinScalarHistory)
    assert isinstance(engine.get_history(f"{ct.EVAL}/{name}_min"), MinScalarHistory)
    assert isinstance(engine.get_history(f"{ct.EVAL}/{name}_sum"), MinScalarHistory)
    assert engine.has_event_handler(
        VanillaEventHandler(metric.reset), EngineEvents.EVAL_EPOCH_STARTED
    )
    assert engine.has_event_handler(
        VanillaEventHandler(metric.value, handler_kwargs={"engine": engine}),
        EngineEvents.EVAL_EPOCH_COMPLETED,
    )


def test_squared_symlog_error_attach_state_mean(engine: BaseEngine) -> None:
    metric = SquaredSymlogError(ct.EVAL, state=MeanErrorState())
    metric.attach(engine)
    assert isinstance(engine.get_history(f"{ct.EVAL}/sq_symlog_err_mean"), MinScalarHistory)
    assert engine.has_event_handler(
        VanillaEventHandler(metric.reset), EngineEvents.EVAL_EPOCH_STARTED
    )
    assert engine.has_event_handler(
        VanillaEventHandler(metric.value, handler_kwargs={"engine": engine}),
        EngineEvents.EVAL_EPOCH_COMPLETED,
    )


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("feature_size", SIZES)
def test_squared_symlog_error_forward_correct(
    device: str, mode: str, batch_size: int, feature_size: int
) -> None:
    device = torch.device(device)
    metric = SquaredSymlogError(mode).to(device=device)
    metric(
        torch.ones(batch_size, feature_size, device=device),
        torch.ones(batch_size, feature_size, device=device),
    )
    assert metric.value() == {
        f"{mode}/sq_symlog_err_mean": 0.0,
        f"{mode}/sq_symlog_err_max": 0.0,
        f"{mode}/sq_symlog_err_min": 0.0,
        f"{mode}/sq_symlog_err_sum": 0.0,
        f"{mode}/sq_symlog_err_num_predictions": batch_size * feature_size,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("feature_size", SIZES)
def test_squared_symlog_error_forward_incorrect(
    device: str, mode: str, batch_size: int, feature_size: int
) -> None:
    device = torch.device(device)
    metric = SquaredSymlogError(mode).to(device=device)
    metric(
        2 * torch.ones(batch_size, feature_size, device=device),
        torch.ones(batch_size, feature_size, device=device),
    )
    assert objects_are_allclose(
        metric.value(),
        {
            f"{mode}/sq_symlog_err_mean": 0.16440195389316548,
            f"{mode}/sq_symlog_err_max": 0.16440195389316548,
            f"{mode}/sq_symlog_err_min": 0.16440195389316548,
            f"{mode}/sq_symlog_err_sum": 0.16440195389316548 * batch_size * feature_size,
            f"{mode}/sq_symlog_err_num_predictions": batch_size * feature_size,
        },
    )


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_squared_symlog_error_forward_incorrect_negative(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = SquaredSymlogError(mode).to(device=device)
    metric(
        -2 * torch.ones(2, 3, device=device),
        -torch.ones(2, 3, device=device),
    )
    assert objects_are_allclose(
        metric.value(),
        {
            f"{mode}/sq_symlog_err_mean": 0.16440195389316548,
            f"{mode}/sq_symlog_err_max": 0.16440195389316548,
            f"{mode}/sq_symlog_err_min": 0.16440195389316548,
            f"{mode}/sq_symlog_err_sum": 0.9864117233589929,
            f"{mode}/sq_symlog_err_num_predictions": 6,
        },
    )


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_squared_symlog_error_forward_partially_correct(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = SquaredSymlogError(mode).to(device=device)
    metric(torch.ones(2, 2, device=device), torch.eye(2, device=device))
    assert objects_are_allclose(
        metric.value(),
        {
            f"{mode}/sq_symlog_err_mean": 0.2402265069591007,
            f"{mode}/sq_symlog_err_max": 0.4804530139182014,
            f"{mode}/sq_symlog_err_min": 0.0,
            f"{mode}/sq_symlog_err_sum": 0.9609060278364028,
            f"{mode}/sq_symlog_err_num_predictions": 4,
        },
    )


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_squared_symlog_error_forward_1d(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = SquaredSymlogError(mode).to(device=device)
    metric(torch.ones(2, device=device), torch.ones(2, device=device))
    assert metric.value() == {
        f"{mode}/sq_symlog_err_mean": 0.0,
        f"{mode}/sq_symlog_err_max": 0.0,
        f"{mode}/sq_symlog_err_min": 0.0,
        f"{mode}/sq_symlog_err_sum": 0.0,
        f"{mode}/sq_symlog_err_num_predictions": 2,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_squared_symlog_error_forward_2d(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = SquaredSymlogError(mode).to(device=device)
    metric(torch.ones(2, 3, device=device), torch.ones(2, 3, device=device))
    assert metric.value() == {
        f"{mode}/sq_symlog_err_mean": 0.0,
        f"{mode}/sq_symlog_err_max": 0.0,
        f"{mode}/sq_symlog_err_min": 0.0,
        f"{mode}/sq_symlog_err_sum": 0.0,
        f"{mode}/sq_symlog_err_num_predictions": 6,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_squared_symlog_error_forward_3d(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = SquaredSymlogError(mode).to(device=device)
    metric(torch.ones(2, 3, 4, device=device), torch.ones(2, 3, 4, device=device))
    assert metric.value() == {
        f"{mode}/sq_symlog_err_mean": 0.0,
        f"{mode}/sq_symlog_err_max": 0.0,
        f"{mode}/sq_symlog_err_min": 0.0,
        f"{mode}/sq_symlog_err_sum": 0.0,
        f"{mode}/sq_symlog_err_num_predictions": 24,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
@mark.parametrize("dtype_prediction", DTYPES)
@mark.parametrize("dtype_target", DTYPES)
def test_squared_symlog_error_forward_dtypes(
    device: str,
    mode: str,
    dtype_prediction: torch.dtype,
    dtype_target: torch.dtype,
) -> None:
    device = torch.device(device)
    metric = SquaredSymlogError(mode).to(device=device)
    metric(
        torch.ones(2, 2, device=device, dtype=dtype_prediction),
        torch.ones(2, 2, device=device, dtype=dtype_target),
    )
    assert metric.value() == {
        f"{mode}/sq_symlog_err_mean": 0.0,
        f"{mode}/sq_symlog_err_max": 0.0,
        f"{mode}/sq_symlog_err_min": 0.0,
        f"{mode}/sq_symlog_err_sum": 0.0,
        f"{mode}/sq_symlog_err_num_predictions": 4,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_squared_symlog_error_forward_state(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = SquaredSymlogError(mode, state=ExtendedErrorState()).to(device=device)
    metric(torch.ones(2, 2, device=device), torch.ones(2, 2, device=device))
    assert metric.value() == {
        f"{mode}/sq_symlog_err_mean": 0.0,
        f"{mode}/sq_symlog_err_max": 0.0,
        f"{mode}/sq_symlog_err_min": 0.0,
        f"{mode}/sq_symlog_err_sum": 0.0,
        f"{mode}/sq_symlog_err_std": 0.0,
        f"{mode}/sq_symlog_err_median": 0.0,
        f"{mode}/sq_symlog_err_num_predictions": 4,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_squared_symlog_error_forward_multiple_batches(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = SquaredSymlogError(mode).to(device=device)
    metric(torch.ones(2, 2, device=device), torch.ones(2, 2, device=device))
    metric(torch.ones(2, 2, device=device), torch.eye(2, device=device))
    assert objects_are_allclose(
        metric.value(),
        {
            f"{mode}/sq_symlog_err_mean": 0.12011325347955035,
            f"{mode}/sq_symlog_err_max": 0.4804530139182014,
            f"{mode}/sq_symlog_err_min": 0.0,
            f"{mode}/sq_symlog_err_sum": 0.9609060278364028,
            f"{mode}/sq_symlog_err_num_predictions": 8,
        },
    )


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_squared_symlog_error_forward_multiple_batches_with_reset(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = SquaredSymlogError(mode).to(device=device)
    metric(torch.ones(2, 2, device=device), torch.ones(2, 2, device=device))
    metric.reset()
    metric(torch.ones(2, 2, device=device), torch.eye(2, device=device))
    assert objects_are_allclose(
        metric.value(),
        {
            f"{mode}/sq_symlog_err_mean": 0.2402265069591007,
            f"{mode}/sq_symlog_err_max": 0.4804530139182014,
            f"{mode}/sq_symlog_err_min": 0.0,
            f"{mode}/sq_symlog_err_sum": 0.9609060278364028,
            f"{mode}/sq_symlog_err_num_predictions": 4,
        },
    )


@mark.parametrize("mode", MODES)
def test_squared_symlog_error_value_empty(mode: str) -> None:
    with raises(EmptyMetricError):
        SquaredSymlogError(mode).value()


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_squared_symlog_error_value_log_engine(device: str, mode: str, engine: BaseEngine) -> None:
    device = torch.device(device)
    metric = SquaredSymlogError(mode).to(device=device)
    metric(torch.ones(2, 2, device=device), torch.ones(2, 2, device=device))
    metric.value(engine)
    assert engine.get_history(f"{mode}/sq_symlog_err_mean").get_last_value() == 0.0
    assert engine.get_history(f"{mode}/sq_symlog_err_max").get_last_value() == 0.0
    assert engine.get_history(f"{mode}/sq_symlog_err_min").get_last_value() == 0.0
    assert engine.get_history(f"{mode}/sq_symlog_err_sum").get_last_value() == 0.0
    assert engine.get_history(f"{mode}/sq_symlog_err_num_predictions").get_last_value() == 4


@mark.parametrize("device", get_available_devices())
def test_squared_symlog_error_events_train(device: str, engine: BaseEngine) -> None:
    device = torch.device(device)
    metric = SquaredSymlogError(ct.TRAIN).to(device=device)
    metric.attach(engine)
    metric(torch.ones(2, 2, device=device), 2 * torch.ones(2, 2, device=device))
    engine.fire_event(EngineEvents.TRAIN_EPOCH_STARTED)
    metric(torch.eye(2, device=device), torch.eye(2, device=device))
    engine.fire_event(EngineEvents.TRAIN_EPOCH_COMPLETED)
    assert engine.get_history(f"{ct.TRAIN}/sq_symlog_err_mean").get_last_value() == 0.0
    assert engine.get_history(f"{ct.TRAIN}/sq_symlog_err_max").get_last_value() == 0.0
    assert engine.get_history(f"{ct.TRAIN}/sq_symlog_err_min").get_last_value() == 0.0
    assert engine.get_history(f"{ct.TRAIN}/sq_symlog_err_sum").get_last_value() == 0.0
    assert engine.get_history(f"{ct.TRAIN}/sq_symlog_err_num_predictions").get_last_value() == 4


@mark.parametrize("device", get_available_devices())
def test_squared_symlog_error_events_eval(device: str, engine: BaseEngine) -> None:
    device = torch.device(device)
    metric = SquaredSymlogError(ct.EVAL).to(device=device)
    metric.attach(engine)
    metric(torch.ones(2, 2, device=device), 2 * torch.ones(2, 2, device=device))
    engine.fire_event(EngineEvents.EVAL_EPOCH_STARTED)
    metric(torch.eye(2, device=device), torch.eye(2, device=device))
    engine.fire_event(EngineEvents.EVAL_EPOCH_COMPLETED)
    assert engine.get_history(f"{ct.EVAL}/sq_symlog_err_mean").get_last_value() == 0.0
    assert engine.get_history(f"{ct.EVAL}/sq_symlog_err_max").get_last_value() == 0.0
    assert engine.get_history(f"{ct.EVAL}/sq_symlog_err_min").get_last_value() == 0.0
    assert engine.get_history(f"{ct.EVAL}/sq_symlog_err_sum").get_last_value() == 0.0
    assert engine.get_history(f"{ct.EVAL}/sq_symlog_err_num_predictions").get_last_value() == 4


def test_squared_symlog_error_reset() -> None:
    state = Mock(spec=BaseState)
    metric = SquaredSymlogError(ct.EVAL, state=state)
    metric.reset()
    state.reset.assert_called_once_with()


#######################################
#     Tests for SquaredAsinhError     #
#######################################


@mark.parametrize("mode", MODES)
def test_squared_asinh_error_str(mode: str) -> None:
    assert str(SquaredAsinhError(mode)).startswith("SquaredAsinhError(")


def test_squared_asinh_error_init_state_default() -> None:
    assert isinstance(SquaredAsinhError(ct.EVAL)._state, ErrorState)


def test_squared_asinh_error_init_state_mean() -> None:
    assert isinstance(SquaredAsinhError(ct.EVAL, state=MeanErrorState())._state, MeanErrorState)


@mark.parametrize("name", NAMES)
def test_squared_asinh_error_attach_train(name: str, engine: BaseEngine) -> None:
    metric = SquaredAsinhError(ct.TRAIN, name=name)
    metric.attach(engine)
    assert isinstance(engine.get_history(f"{ct.TRAIN}/{name}_mean"), MinScalarHistory)
    assert isinstance(engine.get_history(f"{ct.TRAIN}/{name}_max"), MinScalarHistory)
    assert isinstance(engine.get_history(f"{ct.TRAIN}/{name}_min"), MinScalarHistory)
    assert isinstance(engine.get_history(f"{ct.TRAIN}/{name}_sum"), MinScalarHistory)
    assert engine.has_event_handler(
        VanillaEventHandler(metric.reset), EngineEvents.TRAIN_EPOCH_STARTED
    )
    assert engine.has_event_handler(
        VanillaEventHandler(metric.value, handler_kwargs={"engine": engine}),
        EngineEvents.TRAIN_EPOCH_COMPLETED,
    )


@mark.parametrize("name", NAMES)
def test_squared_asinh_error_attach_eval(name: str, engine: BaseEngine) -> None:
    metric = SquaredAsinhError(ct.EVAL, name=name)
    metric.attach(engine)
    assert isinstance(engine.get_history(f"{ct.EVAL}/{name}_mean"), MinScalarHistory)
    assert isinstance(engine.get_history(f"{ct.EVAL}/{name}_max"), MinScalarHistory)
    assert isinstance(engine.get_history(f"{ct.EVAL}/{name}_min"), MinScalarHistory)
    assert isinstance(engine.get_history(f"{ct.EVAL}/{name}_sum"), MinScalarHistory)
    assert engine.has_event_handler(
        VanillaEventHandler(metric.reset), EngineEvents.EVAL_EPOCH_STARTED
    )
    assert engine.has_event_handler(
        VanillaEventHandler(metric.value, handler_kwargs={"engine": engine}),
        EngineEvents.EVAL_EPOCH_COMPLETED,
    )


def test_squared_asinh_error_attach_state_mean(engine: BaseEngine) -> None:
    metric = SquaredAsinhError(ct.EVAL, state=MeanErrorState())
    metric.attach(engine)
    assert isinstance(engine.get_history(f"{ct.EVAL}/sq_asinh_err_mean"), MinScalarHistory)
    assert engine.has_event_handler(
        VanillaEventHandler(metric.reset), EngineEvents.EVAL_EPOCH_STARTED
    )
    assert engine.has_event_handler(
        VanillaEventHandler(metric.value, handler_kwargs={"engine": engine}),
        EngineEvents.EVAL_EPOCH_COMPLETED,
    )


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("feature_size", SIZES)
def test_squared_asinh_error_forward_correct(
    device: str, mode: str, batch_size: int, feature_size: int
) -> None:
    device = torch.device(device)
    metric = SquaredAsinhError(mode).to(device=device)
    metric(
        torch.ones(batch_size, feature_size, device=device),
        torch.ones(batch_size, feature_size, device=device),
    )
    assert metric.value() == {
        f"{mode}/sq_asinh_err_mean": 0.0,
        f"{mode}/sq_asinh_err_max": 0.0,
        f"{mode}/sq_asinh_err_min": 0.0,
        f"{mode}/sq_asinh_err_sum": 0.0,
        f"{mode}/sq_asinh_err_num_predictions": batch_size * feature_size,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("feature_size", SIZES)
def test_squared_asinh_error_forward_incorrect(
    device: str, mode: str, batch_size: int, feature_size: int
) -> None:
    device = torch.device(device)
    metric = SquaredAsinhError(mode).to(device=device)
    metric(
        2 * torch.ones(batch_size, feature_size, device=device),
        torch.ones(batch_size, feature_size, device=device),
    )
    assert objects_are_allclose(
        metric.value(),
        {
            f"{mode}/sq_asinh_err_mean": 0.31613843087642435,
            f"{mode}/sq_asinh_err_max": 0.31613843087642435,
            f"{mode}/sq_asinh_err_min": 0.31613843087642435,
            f"{mode}/sq_asinh_err_sum": 0.31613843087642435 * batch_size * feature_size,
            f"{mode}/sq_asinh_err_num_predictions": batch_size * feature_size,
        },
    )


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_squared_asinh_error_forward_incorrect_negative(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = SquaredAsinhError(mode).to(device=device)
    metric(
        -2 * torch.ones(2, 3, device=device),
        -torch.ones(2, 3, device=device),
    )
    assert objects_are_allclose(
        metric.value(),
        {
            f"{mode}/sq_asinh_err_mean": 0.31613843087642435,
            f"{mode}/sq_asinh_err_max": 0.31613843087642435,
            f"{mode}/sq_asinh_err_min": 0.31613843087642435,
            f"{mode}/sq_asinh_err_sum": 1.896830585258546,
            f"{mode}/sq_asinh_err_num_predictions": 6,
        },
    )


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_squared_asinh_error_forward_partially_correct(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = SquaredAsinhError(mode).to(device=device)
    metric(torch.ones(2, 2, device=device), torch.eye(2, device=device))
    assert objects_are_allclose(
        metric.value(),
        {
            f"{mode}/sq_asinh_err_mean": 0.388409699947848,
            f"{mode}/sq_asinh_err_max": 0.776819399895696,
            f"{mode}/sq_asinh_err_min": 0.0,
            f"{mode}/sq_asinh_err_sum": 1.553638799791392,
            f"{mode}/sq_asinh_err_num_predictions": 4,
        },
    )


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_squared_asinh_error_forward_1d(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = SquaredAsinhError(mode).to(device=device)
    metric(torch.ones(2, device=device), torch.ones(2, device=device))
    assert metric.value() == {
        f"{mode}/sq_asinh_err_mean": 0.0,
        f"{mode}/sq_asinh_err_max": 0.0,
        f"{mode}/sq_asinh_err_min": 0.0,
        f"{mode}/sq_asinh_err_sum": 0.0,
        f"{mode}/sq_asinh_err_num_predictions": 2,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_squared_asinh_error_forward_2d(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = SquaredAsinhError(mode).to(device=device)
    metric(torch.ones(2, 3, device=device), torch.ones(2, 3, device=device))
    assert metric.value() == {
        f"{mode}/sq_asinh_err_mean": 0.0,
        f"{mode}/sq_asinh_err_max": 0.0,
        f"{mode}/sq_asinh_err_min": 0.0,
        f"{mode}/sq_asinh_err_sum": 0.0,
        f"{mode}/sq_asinh_err_num_predictions": 6,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_squared_asinh_error_forward_3d(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = SquaredAsinhError(mode).to(device=device)
    metric(torch.ones(2, 3, 4, device=device), torch.ones(2, 3, 4, device=device))
    assert metric.value() == {
        f"{mode}/sq_asinh_err_mean": 0.0,
        f"{mode}/sq_asinh_err_max": 0.0,
        f"{mode}/sq_asinh_err_min": 0.0,
        f"{mode}/sq_asinh_err_sum": 0.0,
        f"{mode}/sq_asinh_err_num_predictions": 24,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
@mark.parametrize("dtype_prediction", DTYPES)
@mark.parametrize("dtype_target", DTYPES)
def test_squared_asinh_error_forward_dtypes(
    device: str,
    mode: str,
    dtype_prediction: torch.dtype,
    dtype_target: torch.dtype,
) -> None:
    device = torch.device(device)
    metric = SquaredAsinhError(mode).to(device=device)
    metric(
        torch.ones(2, 2, device=device, dtype=dtype_prediction),
        torch.ones(2, 2, device=device, dtype=dtype_target),
    )
    assert metric.value() == {
        f"{mode}/sq_asinh_err_mean": 0.0,
        f"{mode}/sq_asinh_err_max": 0.0,
        f"{mode}/sq_asinh_err_min": 0.0,
        f"{mode}/sq_asinh_err_sum": 0.0,
        f"{mode}/sq_asinh_err_num_predictions": 4,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_squared_asinh_error_forward_state(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = SquaredAsinhError(mode, state=ExtendedErrorState()).to(device=device)
    metric(torch.ones(2, 2, device=device), torch.ones(2, 2, device=device))
    assert metric.value() == {
        f"{mode}/sq_asinh_err_mean": 0.0,
        f"{mode}/sq_asinh_err_max": 0.0,
        f"{mode}/sq_asinh_err_min": 0.0,
        f"{mode}/sq_asinh_err_sum": 0.0,
        f"{mode}/sq_asinh_err_std": 0.0,
        f"{mode}/sq_asinh_err_median": 0.0,
        f"{mode}/sq_asinh_err_num_predictions": 4,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_squared_asinh_error_forward_multiple_batches(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = SquaredAsinhError(mode).to(device=device)
    metric(torch.ones(2, 2, device=device), torch.ones(2, 2, device=device))
    metric(torch.ones(2, 2, device=device), torch.eye(2, device=device))
    assert objects_are_allclose(
        metric.value(),
        {
            f"{mode}/sq_asinh_err_mean": 0.194204849973924,
            f"{mode}/sq_asinh_err_max": 0.776819399895696,
            f"{mode}/sq_asinh_err_min": 0.0,
            f"{mode}/sq_asinh_err_sum": 1.553638799791392,
            f"{mode}/sq_asinh_err_num_predictions": 8,
        },
    )


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_squared_asinh_error_forward_multiple_batches_with_reset(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = SquaredAsinhError(mode).to(device=device)
    metric(torch.ones(2, 2, device=device), torch.ones(2, 2, device=device))
    metric.reset()
    metric(torch.ones(2, 2, device=device), torch.eye(2, device=device))
    assert objects_are_allclose(
        metric.value(),
        {
            f"{mode}/sq_asinh_err_mean": 0.388409699947848,
            f"{mode}/sq_asinh_err_max": 0.776819399895696,
            f"{mode}/sq_asinh_err_min": 0.0,
            f"{mode}/sq_asinh_err_sum": 1.553638799791392,
            f"{mode}/sq_asinh_err_num_predictions": 4,
        },
    )


@mark.parametrize("mode", MODES)
def test_squared_asinh_error_value_empty(mode: str) -> None:
    with raises(EmptyMetricError):
        SquaredAsinhError(mode).value()


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_squared_asinh_error_value_log_engine(device: str, mode: str, engine: BaseEngine) -> None:
    device = torch.device(device)
    metric = SquaredAsinhError(mode).to(device=device)
    metric(torch.ones(2, 2, device=device), torch.ones(2, 2, device=device))
    metric.value(engine)
    assert engine.get_history(f"{mode}/sq_asinh_err_mean").get_last_value() == 0.0
    assert engine.get_history(f"{mode}/sq_asinh_err_max").get_last_value() == 0.0
    assert engine.get_history(f"{mode}/sq_asinh_err_min").get_last_value() == 0.0
    assert engine.get_history(f"{mode}/sq_asinh_err_sum").get_last_value() == 0.0
    assert engine.get_history(f"{mode}/sq_asinh_err_num_predictions").get_last_value() == 4


@mark.parametrize("device", get_available_devices())
def test_squared_asinh_error_events_train(device: str, engine: BaseEngine) -> None:
    device = torch.device(device)
    metric = SquaredAsinhError(ct.TRAIN).to(device=device)
    metric.attach(engine)
    metric(torch.ones(2, 2, device=device), 2 * torch.ones(2, 2, device=device))
    engine.fire_event(EngineEvents.TRAIN_EPOCH_STARTED)
    metric(torch.eye(2, device=device), torch.eye(2, device=device))
    engine.fire_event(EngineEvents.TRAIN_EPOCH_COMPLETED)
    assert engine.get_history(f"{ct.TRAIN}/sq_asinh_err_mean").get_last_value() == 0.0
    assert engine.get_history(f"{ct.TRAIN}/sq_asinh_err_max").get_last_value() == 0.0
    assert engine.get_history(f"{ct.TRAIN}/sq_asinh_err_min").get_last_value() == 0.0
    assert engine.get_history(f"{ct.TRAIN}/sq_asinh_err_sum").get_last_value() == 0.0
    assert engine.get_history(f"{ct.TRAIN}/sq_asinh_err_num_predictions").get_last_value() == 4


@mark.parametrize("device", get_available_devices())
def test_squared_asinh_error_events_eval(device: str, engine: BaseEngine) -> None:
    device = torch.device(device)
    metric = SquaredAsinhError(ct.EVAL).to(device=device)
    metric.attach(engine)
    metric(torch.ones(2, 2, device=device), 2 * torch.ones(2, 2, device=device))
    engine.fire_event(EngineEvents.EVAL_EPOCH_STARTED)
    metric(torch.eye(2, device=device), torch.eye(2, device=device))
    engine.fire_event(EngineEvents.EVAL_EPOCH_COMPLETED)
    assert engine.get_history(f"{ct.EVAL}/sq_asinh_err_mean").get_last_value() == 0.0
    assert engine.get_history(f"{ct.EVAL}/sq_asinh_err_max").get_last_value() == 0.0
    assert engine.get_history(f"{ct.EVAL}/sq_asinh_err_min").get_last_value() == 0.0
    assert engine.get_history(f"{ct.EVAL}/sq_asinh_err_sum").get_last_value() == 0.0
    assert engine.get_history(f"{ct.EVAL}/sq_asinh_err_num_predictions").get_last_value() == 4


def test_squared_asinh_error_reset() -> None:
    state = Mock(spec=BaseState)
    metric = SquaredAsinhError(ct.EVAL, state=state)
    metric.reset()
    state.reset.assert_called_once_with()
