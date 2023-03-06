from unittest.mock import Mock

import torch
from pytest import fixture, mark, raises

from gravitorch import constants as ct
from gravitorch.engines import BaseEngine, EngineEvents
from gravitorch.models.metrics import (
    AbsoluteRelativeError,
    EmptyMetricError,
    SymmetricAbsoluteRelativeError,
)
from gravitorch.models.metrics.state import (
    BaseState,
    ErrorState,
    ExtendedErrorState,
    MeanErrorState,
)
from gravitorch.testing import create_dummy_engine
from gravitorch.utils import get_available_devices
from gravitorch.utils.events import VanillaEventHandler
from gravitorch.utils.history import MinScalarHistory

MODES = (ct.TRAIN, ct.EVAL)
NAMES = ("abs_rel_err", "absolute_relative_error")
SIZES = (1, 2)
DTYPES = (torch.long, torch.float)


@fixture
def engine() -> BaseEngine:
    return create_dummy_engine()


###########################################
#     Tests for AbsoluteRelativeError     #
###########################################


@mark.parametrize("mode", MODES)
def test_absolute_relative_error_str(mode: str):
    assert str(AbsoluteRelativeError(mode)).startswith("AbsoluteRelativeError(")


@mark.parametrize("eps", (1e-5, 0.1))
def test_absolute_relative_error_init_eps(eps: float):
    assert AbsoluteRelativeError(mode=ct.EVAL, eps=eps)._eps == eps


def test_absolute_relative_error_init_eps_default() -> None:
    assert AbsoluteRelativeError(mode=ct.EVAL)._eps == 1e-8


@mark.parametrize("eps", (0.0, -0.1))
def test_absolute_relative_error_init_eps_incorrect(eps: float):
    with raises(ValueError):
        AbsoluteRelativeError(mode=ct.EVAL, eps=eps)


def test_absolute_relative_error_init_state_default() -> None:
    assert isinstance(AbsoluteRelativeError(ct.EVAL)._state, ErrorState)


def test_absolute_relative_error_init_state_mean() -> None:
    assert isinstance(AbsoluteRelativeError(ct.EVAL, state=MeanErrorState())._state, MeanErrorState)


@mark.parametrize("name", NAMES)
def test_absolute_relative_error_attach_train(name: str, engine: BaseEngine):
    metric = AbsoluteRelativeError(ct.TRAIN, name=name)
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
def test_absolute_relative_error_attach_eval(name: str, engine: BaseEngine):
    metric = AbsoluteRelativeError(ct.EVAL, name=name)
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


def test_absolute_relative_error_attach_state_mean(engine: BaseEngine):
    metric = AbsoluteRelativeError(ct.EVAL, state=MeanErrorState())
    metric.attach(engine)
    assert isinstance(engine.get_history(f"{ct.EVAL}/abs_rel_err_mean"), MinScalarHistory)
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
def test_absolute_relative_error_forward_correct(
    device: str, mode: str, batch_size: int, feature_size: int
):
    device = torch.device(device)
    metric = AbsoluteRelativeError(mode).to(device=device)
    metric(
        torch.ones(batch_size, feature_size, device=device),
        torch.ones(batch_size, feature_size, device=device),
    )
    assert metric.value() == {
        f"{mode}/abs_rel_err_mean": 0.0,
        f"{mode}/abs_rel_err_max": 0.0,
        f"{mode}/abs_rel_err_min": 0.0,
        f"{mode}/abs_rel_err_sum": 0.0,
        f"{mode}/abs_rel_err_num_predictions": batch_size * feature_size,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_absolute_relative_error_forward_correct_zero(device: str, mode: str):
    device = torch.device(device)
    metric = AbsoluteRelativeError(mode).to(device=device)
    metric(torch.zeros(2, 2, device=device), torch.zeros(2, 2, device=device))
    assert metric.value() == {
        f"{mode}/abs_rel_err_mean": 0.0,
        f"{mode}/abs_rel_err_max": 0.0,
        f"{mode}/abs_rel_err_min": 0.0,
        f"{mode}/abs_rel_err_sum": 0.0,
        f"{mode}/abs_rel_err_num_predictions": 4,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("feature_size", SIZES)
def test_absolute_relative_error_forward_incorrect(
    device: str, mode: str, batch_size: int, feature_size: int
):
    device = torch.device(device)
    metric = AbsoluteRelativeError(mode).to(device=device)
    metric(
        torch.ones(batch_size, feature_size, device=device).mul(2),
        torch.ones(batch_size, feature_size, device=device),
    )
    assert metric.value() == {
        f"{mode}/abs_rel_err_mean": 1.0,
        f"{mode}/abs_rel_err_max": 1.0,
        f"{mode}/abs_rel_err_min": 1.0,
        f"{mode}/abs_rel_err_sum": batch_size * feature_size,
        f"{mode}/abs_rel_err_num_predictions": batch_size * feature_size,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_absolute_relative_error_forward_partially_correct_zero_prediction(device: str, mode: str):
    device = torch.device(device)
    metric = AbsoluteRelativeError(mode).to(device=device)
    metric(torch.eye(2, device=device), torch.ones(2, 2, device=device))
    assert metric.value() == {
        f"{mode}/abs_rel_err_mean": 0.5,
        f"{mode}/abs_rel_err_max": 1.0,
        f"{mode}/abs_rel_err_min": 0.0,
        f"{mode}/abs_rel_err_sum": 2.0,
        f"{mode}/abs_rel_err_num_predictions": 4,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_absolute_relative_error_forward_partially_correct_zero_target(device: str, mode: str):
    device = torch.device(device)
    metric = AbsoluteRelativeError(mode).to(device=device)
    metric(torch.ones(2, 2, device=device), torch.eye(2, device=device))
    assert metric.value() == {
        f"{mode}/abs_rel_err_mean": 5e7,
        f"{mode}/abs_rel_err_max": 1e8,
        f"{mode}/abs_rel_err_min": 0.0,
        f"{mode}/abs_rel_err_sum": 2e8,
        f"{mode}/abs_rel_err_num_predictions": 4,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_absolute_relative_error_forward_1d(device: str, mode: str):
    device = torch.device(device)
    metric = AbsoluteRelativeError(mode).to(device=device)
    metric(torch.ones(2, device=device), torch.ones(2, device=device))
    assert metric.value() == {
        f"{mode}/abs_rel_err_mean": 0.0,
        f"{mode}/abs_rel_err_max": 0.0,
        f"{mode}/abs_rel_err_min": 0.0,
        f"{mode}/abs_rel_err_sum": 0.0,
        f"{mode}/abs_rel_err_num_predictions": 2,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_absolute_relative_error_forward_3d(device: str, mode: str):
    device = torch.device(device)
    metric = AbsoluteRelativeError(mode).to(device=device)
    metric(torch.ones(2, 3, 4, device=device), torch.ones(2, 3, 4, device=device))
    assert metric.value() == {
        f"{mode}/abs_rel_err_mean": 0.0,
        f"{mode}/abs_rel_err_max": 0.0,
        f"{mode}/abs_rel_err_min": 0.0,
        f"{mode}/abs_rel_err_sum": 0.0,
        f"{mode}/abs_rel_err_num_predictions": 24,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
@mark.parametrize("dtype_prediction", DTYPES)
@mark.parametrize("dtype_target", DTYPES)
def test_absolute_relative_error_forward_dtype(
    device: str,
    mode: str,
    dtype_prediction: torch.dtype,
    dtype_target: torch.dtype,
):
    device = torch.device(device)
    metric = AbsoluteRelativeError(mode).to(device=device)
    metric(
        torch.ones(2, 3, device=device, dtype=dtype_prediction),
        torch.ones(2, 3, device=device, dtype=dtype_target),
    )
    assert metric.value() == {
        f"{mode}/abs_rel_err_mean": 0.0,
        f"{mode}/abs_rel_err_max": 0.0,
        f"{mode}/abs_rel_err_min": 0.0,
        f"{mode}/abs_rel_err_sum": 0.0,
        f"{mode}/abs_rel_err_num_predictions": 6,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_absolute_relative_error_forward_state(device: str, mode: str):
    device = torch.device(device)
    metric = AbsoluteRelativeError(mode, state=ExtendedErrorState()).to(device=device)
    metric(torch.ones(2, 2, device=device), torch.ones(2, 2, device=device))
    assert metric.value() == {
        f"{mode}/abs_rel_err_mean": 0.0,
        f"{mode}/abs_rel_err_max": 0.0,
        f"{mode}/abs_rel_err_min": 0.0,
        f"{mode}/abs_rel_err_sum": 0.0,
        f"{mode}/abs_rel_err_std": 0.0,
        f"{mode}/abs_rel_err_median": 0.0,
        f"{mode}/abs_rel_err_num_predictions": 4,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_absolute_relative_error_forward_multiple_batches(device: str, mode: str):
    device = torch.device(device)
    metric = AbsoluteRelativeError(mode).to(device=device)
    metric(torch.ones(2, 2, device=device), torch.ones(2, 2, device=device))
    metric(torch.ones(2, 2, device=device), torch.eye(2, device=device) + 1)
    assert metric.value() == {
        f"{mode}/abs_rel_err_mean": 0.125,
        f"{mode}/abs_rel_err_max": 0.5,
        f"{mode}/abs_rel_err_min": 0.0,
        f"{mode}/abs_rel_err_sum": 1.0,
        f"{mode}/abs_rel_err_num_predictions": 8,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_absolute_relative_error_forward_multiple_batches_with_reset(device: str, mode: str):
    device = torch.device(device)
    metric = AbsoluteRelativeError(mode).to(device=device)
    metric(torch.ones(2, 2, device=device), torch.ones(2, 2, device=device))
    metric.reset()
    metric(torch.ones(2, 2, device=device), torch.eye(2, device=device) + 1)
    assert metric.value() == {
        f"{mode}/abs_rel_err_mean": 0.25,
        f"{mode}/abs_rel_err_max": 0.5,
        f"{mode}/abs_rel_err_min": 0.0,
        f"{mode}/abs_rel_err_sum": 1.0,
        f"{mode}/abs_rel_err_num_predictions": 4,
    }


@mark.parametrize("mode", MODES)
def test_absolute_relative_error_value_empty(mode: str):
    with raises(EmptyMetricError):
        AbsoluteRelativeError(mode).value()


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_absolute_relative_error_value_log_engine(device: str, mode: str, engine: BaseEngine):
    device = torch.device(device)
    metric = AbsoluteRelativeError(mode).to(device=device)
    metric(torch.ones(2, 2, device=device), torch.ones(2, 2, device=device))
    metric.value(engine)
    assert engine.get_history(f"{mode}/abs_rel_err_mean").get_last_value() == 0.0
    assert engine.get_history(f"{mode}/abs_rel_err_max").get_last_value() == 0.0
    assert engine.get_history(f"{mode}/abs_rel_err_min").get_last_value() == 0.0
    assert engine.get_history(f"{mode}/abs_rel_err_sum").get_last_value() == 0.0
    assert engine.get_history(f"{mode}/abs_rel_err_num_predictions").get_last_value() == 4


@mark.parametrize("device", get_available_devices())
def test_absolute_relative_error_events_train(device: str, engine: BaseEngine):
    device = torch.device(device)
    metric = AbsoluteRelativeError(ct.TRAIN).to(device=device)
    metric.attach(engine)
    metric(torch.ones(2, 2, device=device), 2 * torch.ones(2, 2, device=device))
    engine.fire_event(EngineEvents.TRAIN_EPOCH_STARTED)
    metric(torch.eye(2, device=device) + 1, torch.eye(2, device=device) + 1)
    engine.fire_event(EngineEvents.TRAIN_EPOCH_COMPLETED)
    assert engine.get_history(f"{ct.TRAIN}/abs_rel_err_mean").get_last_value() == 0.0
    assert engine.get_history(f"{ct.TRAIN}/abs_rel_err_max").get_last_value() == 0.0
    assert engine.get_history(f"{ct.TRAIN}/abs_rel_err_min").get_last_value() == 0.0
    assert engine.get_history(f"{ct.TRAIN}/abs_rel_err_sum").get_last_value() == 0.0
    assert engine.get_history(f"{ct.TRAIN}/abs_rel_err_num_predictions").get_last_value() == 4


@mark.parametrize("device", get_available_devices())
def test_absolute_relative_error_events_eval(device: str, engine: BaseEngine):
    device = torch.device(device)
    metric = AbsoluteRelativeError(ct.EVAL).to(device=device)
    metric.attach(engine)
    metric(torch.ones(2, 2, device=device), 2 * torch.ones(2, 2, device=device))
    engine.fire_event(EngineEvents.EVAL_EPOCH_STARTED)
    metric(torch.eye(2, device=device) + 1, torch.eye(2, device=device) + 1)
    engine.fire_event(EngineEvents.EVAL_EPOCH_COMPLETED)
    assert engine.get_history(f"{ct.EVAL}/abs_rel_err_mean").get_last_value() == 0.0
    assert engine.get_history(f"{ct.EVAL}/abs_rel_err_max").get_last_value() == 0.0
    assert engine.get_history(f"{ct.EVAL}/abs_rel_err_min").get_last_value() == 0.0
    assert engine.get_history(f"{ct.EVAL}/abs_rel_err_sum").get_last_value() == 0.0
    assert engine.get_history(f"{ct.EVAL}/abs_rel_err_num_predictions").get_last_value() == 4


def test_absolute_relative_error_reset() -> None:
    state = Mock(spec=BaseState)
    metric = AbsoluteRelativeError(ct.EVAL, state=state)
    metric.reset()
    state.reset.assert_called_once_with()


####################################################
#     Tests for SymmetricAbsoluteRelativeError     #
####################################################


@mark.parametrize("mode", MODES)
def test_symmetric_absolute_relative_error_str(mode: str):
    assert str(SymmetricAbsoluteRelativeError(mode)).startswith("SymmetricAbsoluteRelativeError(")


@mark.parametrize("eps", (1e-5, 0.1))
def test_symmetric_absolute_relative_error_eps(eps: float):
    assert SymmetricAbsoluteRelativeError(mode=ct.EVAL, eps=eps)._eps == eps


def test_symmetric_absolute_relative_error_eps_default() -> None:
    assert SymmetricAbsoluteRelativeError(mode=ct.EVAL)._eps == 1e-8


@mark.parametrize("eps", (0.0, -0.1))
def test_symmetric_absolute_relative_error_eps_incorrect(eps: float):
    with raises(ValueError):
        SymmetricAbsoluteRelativeError(mode=ct.EVAL, eps=eps)


def test_symmetric_absolute_relative_error_init_state_default() -> None:
    assert isinstance(SymmetricAbsoluteRelativeError(ct.EVAL)._state, ErrorState)


def test_symmetric_absolute_relative_error_init_state_mean() -> None:
    assert isinstance(
        SymmetricAbsoluteRelativeError(ct.EVAL, state=MeanErrorState())._state, MeanErrorState
    )


@mark.parametrize("name", NAMES)
def test_symmetric_absolute_relative_error_attach_train(name: str, engine: BaseEngine):
    metric = SymmetricAbsoluteRelativeError(ct.TRAIN, name=name)
    metric.attach(engine)
    assert isinstance(engine.get_history(f"{ct.TRAIN}/{name}_mean"), MinScalarHistory)
    assert isinstance(engine.get_history(f"{ct.TRAIN}/{name}_max"), MinScalarHistory)
    assert isinstance(engine.get_history(f"{ct.TRAIN}/{name}_min"), MinScalarHistory)
    assert engine.has_event_handler(
        VanillaEventHandler(metric.reset), EngineEvents.TRAIN_EPOCH_STARTED
    )
    assert engine.has_event_handler(
        VanillaEventHandler(metric.value, handler_kwargs={"engine": engine}),
        EngineEvents.TRAIN_EPOCH_COMPLETED,
    )


@mark.parametrize("name", NAMES)
def test_symmetric_absolute_relative_error_attach_eval(name: str, engine: BaseEngine):
    metric = SymmetricAbsoluteRelativeError(ct.EVAL, name=name)
    metric.attach(engine)
    assert isinstance(engine.get_history(f"{ct.EVAL}/{name}_mean"), MinScalarHistory)
    assert isinstance(engine.get_history(f"{ct.EVAL}/{name}_max"), MinScalarHistory)
    assert isinstance(engine.get_history(f"{ct.EVAL}/{name}_min"), MinScalarHistory)
    assert engine.has_event_handler(
        VanillaEventHandler(metric.reset), EngineEvents.EVAL_EPOCH_STARTED
    )
    assert engine.has_event_handler(
        VanillaEventHandler(metric.value, handler_kwargs={"engine": engine}),
        EngineEvents.EVAL_EPOCH_COMPLETED,
    )


def test_symmetric_absolute_relative_error_attach_state_mean(engine: BaseEngine):
    metric = SymmetricAbsoluteRelativeError(ct.EVAL, state=MeanErrorState())
    metric.attach(engine)
    assert isinstance(engine.get_history(f"{ct.EVAL}/sym_abs_rel_err_mean"), MinScalarHistory)
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
def test_symmetric_absolute_relative_error_forward_correct(
    device: str, mode: str, batch_size: int, feature_size: int
):
    device = torch.device(device)
    metric = SymmetricAbsoluteRelativeError(mode).to(device=device)
    metric(
        torch.ones(batch_size, feature_size, device=device),
        torch.ones(batch_size, feature_size, device=device),
    )
    assert metric.value() == {
        f"{mode}/sym_abs_rel_err_mean": 0.0,
        f"{mode}/sym_abs_rel_err_max": 0.0,
        f"{mode}/sym_abs_rel_err_min": 0.0,
        f"{mode}/sym_abs_rel_err_sum": 0.0,
        f"{mode}/sym_abs_rel_err_num_predictions": batch_size * feature_size,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_symmetric_absolute_relative_error_forward_correct_zero(device: str, mode: str):
    device = torch.device(device)
    metric = SymmetricAbsoluteRelativeError(mode).to(device=device)
    metric(torch.zeros(2, 2, device=device), torch.zeros(2, 2, device=device))
    assert metric.value() == {
        f"{mode}/sym_abs_rel_err_mean": 0.0,
        f"{mode}/sym_abs_rel_err_max": 0.0,
        f"{mode}/sym_abs_rel_err_min": 0.0,
        f"{mode}/sym_abs_rel_err_sum": 0.0,
        f"{mode}/sym_abs_rel_err_num_predictions": 4,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("feature_size", SIZES)
def test_symmetric_absolute_relative_error_forward_incorrect(
    device: str, mode: str, batch_size: int, feature_size: int
):
    device = torch.device(device)
    metric = SymmetricAbsoluteRelativeError(mode).to(device=device)
    metric(
        torch.ones(batch_size, feature_size, device=device).mul(3),
        torch.ones(batch_size, feature_size, device=device),
    )
    assert metric.value() == {
        f"{mode}/sym_abs_rel_err_mean": 1.0,
        f"{mode}/sym_abs_rel_err_max": 1.0,
        f"{mode}/sym_abs_rel_err_min": 1.0,
        f"{mode}/sym_abs_rel_err_sum": batch_size * feature_size,
        f"{mode}/sym_abs_rel_err_num_predictions": batch_size * feature_size,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_symmetric_absolute_relative_error_forward_partially_correct_zero_prediction(
    device: str, mode: str
):
    device = torch.device(device)
    metric = SymmetricAbsoluteRelativeError(mode).to(device=device)
    metric(torch.eye(2, device=device), torch.ones(2, 2, device=device))
    assert metric.value() == {
        f"{mode}/sym_abs_rel_err_mean": 1.0,
        f"{mode}/sym_abs_rel_err_max": 2.0,
        f"{mode}/sym_abs_rel_err_min": 0.0,
        f"{mode}/sym_abs_rel_err_sum": 4.0,
        f"{mode}/sym_abs_rel_err_num_predictions": 4,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_symmetric_absolute_relative_error_forward_partially_correct_zero_target(
    device: str, mode: str
):
    device = torch.device(device)
    metric = SymmetricAbsoluteRelativeError(mode).to(device=device)
    metric(torch.ones(2, 2, device=device), torch.eye(2, device=device))
    assert metric.value() == {
        f"{mode}/sym_abs_rel_err_mean": 1.0,
        f"{mode}/sym_abs_rel_err_max": 2.0,
        f"{mode}/sym_abs_rel_err_min": 0.0,
        f"{mode}/sym_abs_rel_err_sum": 4.0,
        f"{mode}/sym_abs_rel_err_num_predictions": 4,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_symmetric_absolute_relative_error_forward_1d(device: str, mode: str):
    device = torch.device(device)
    metric = SymmetricAbsoluteRelativeError(mode).to(device=device)
    metric(torch.ones(2, device=device), torch.ones(2, device=device))
    assert metric.value() == {
        f"{mode}/sym_abs_rel_err_mean": 0.0,
        f"{mode}/sym_abs_rel_err_max": 0.0,
        f"{mode}/sym_abs_rel_err_min": 0.0,
        f"{mode}/sym_abs_rel_err_sum": 0.0,
        f"{mode}/sym_abs_rel_err_num_predictions": 2,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_symmetric_absolute_relative_error_forward_2d(device: str, mode: str):
    device = torch.device(device)
    metric = SymmetricAbsoluteRelativeError(mode).to(device=device)
    metric(torch.ones(2, 3, device=device), torch.ones(2, 3, device=device))
    assert metric.value() == {
        f"{mode}/sym_abs_rel_err_mean": 0.0,
        f"{mode}/sym_abs_rel_err_max": 0.0,
        f"{mode}/sym_abs_rel_err_min": 0.0,
        f"{mode}/sym_abs_rel_err_sum": 0.0,
        f"{mode}/sym_abs_rel_err_num_predictions": 6,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_symmetric_absolute_relative_error_forward_3d(device: str, mode: str):
    device = torch.device(device)
    metric = SymmetricAbsoluteRelativeError(mode).to(device=device)
    metric(torch.ones(2, 3, 4, device=device), torch.ones(2, 3, 4, device=device))
    assert metric.value() == {
        f"{mode}/sym_abs_rel_err_mean": 0.0,
        f"{mode}/sym_abs_rel_err_max": 0.0,
        f"{mode}/sym_abs_rel_err_min": 0.0,
        f"{mode}/sym_abs_rel_err_sum": 0.0,
        f"{mode}/sym_abs_rel_err_num_predictions": 24,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
@mark.parametrize("dtype_prediction", DTYPES)
@mark.parametrize("dtype_target", DTYPES)
def test_symmetric_absolute_relative_error_forward_dtype(
    device: str,
    mode: str,
    dtype_prediction: torch.dtype,
    dtype_target: torch.dtype,
):
    device = torch.device(device)
    metric = SymmetricAbsoluteRelativeError(mode).to(device=device)
    metric(
        torch.ones(2, 3, device=device, dtype=dtype_prediction),
        torch.ones(2, 3, device=device, dtype=dtype_target),
    )
    assert metric.value() == {
        f"{mode}/sym_abs_rel_err_mean": 0.0,
        f"{mode}/sym_abs_rel_err_max": 0.0,
        f"{mode}/sym_abs_rel_err_min": 0.0,
        f"{mode}/sym_abs_rel_err_sum": 0.0,
        f"{mode}/sym_abs_rel_err_num_predictions": 6,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_symmetric_absolute_relative_error_forward_state(device: str, mode: str):
    device = torch.device(device)
    metric = SymmetricAbsoluteRelativeError(mode, state=ExtendedErrorState()).to(device=device)
    metric(torch.ones(2, 2, device=device), torch.ones(2, 2, device=device))
    assert metric.value() == {
        f"{mode}/sym_abs_rel_err_mean": 0.0,
        f"{mode}/sym_abs_rel_err_max": 0.0,
        f"{mode}/sym_abs_rel_err_min": 0.0,
        f"{mode}/sym_abs_rel_err_sum": 0.0,
        f"{mode}/sym_abs_rel_err_std": 0.0,
        f"{mode}/sym_abs_rel_err_median": 0.0,
        f"{mode}/sym_abs_rel_err_num_predictions": 4,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_symmetric_absolute_relative_error_forward_multiple_batches(device: str, mode: str):
    device = torch.device(device)
    metric = SymmetricAbsoluteRelativeError(mode).to(device=device)
    metric(torch.ones(2, 2, device=device), torch.ones(2, 2, device=device))
    metric(torch.ones(2, 2, device=device), torch.eye(2, device=device))
    assert metric.value() == {
        f"{mode}/sym_abs_rel_err_mean": 0.5,
        f"{mode}/sym_abs_rel_err_max": 2.0,
        f"{mode}/sym_abs_rel_err_min": 0.0,
        f"{mode}/sym_abs_rel_err_sum": 4.0,
        f"{mode}/sym_abs_rel_err_num_predictions": 8,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_symmetric_absolute_relative_error_forward_multiple_batches_with_reset(
    device: str, mode: str
):
    device = torch.device(device)
    metric = SymmetricAbsoluteRelativeError(mode).to(device=device)
    metric(torch.ones(2, 2, device=device), torch.ones(2, 2, device=device))
    metric.reset()
    metric(torch.ones(2, 2, device=device), torch.eye(2, device=device))
    assert metric.value() == {
        f"{mode}/sym_abs_rel_err_mean": 1.0,
        f"{mode}/sym_abs_rel_err_max": 2.0,
        f"{mode}/sym_abs_rel_err_min": 0.0,
        f"{mode}/sym_abs_rel_err_sum": 4.0,
        f"{mode}/sym_abs_rel_err_num_predictions": 4,
    }


@mark.parametrize("mode", MODES)
def test_symmetric_absolute_relative_error_value_empty(mode: str):
    with raises(EmptyMetricError):
        SymmetricAbsoluteRelativeError(mode).value()


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_symmetric_absolute_relative_error_value_log_engine(
    device: str, mode: str, engine: BaseEngine
):
    device = torch.device(device)
    metric = SymmetricAbsoluteRelativeError(mode).to(device=device)
    metric(torch.ones(2, 2, device=device), torch.ones(2, 2, device=device))
    metric.value(engine)
    assert engine.get_history(f"{mode}/sym_abs_rel_err_mean").get_last_value() == 0.0
    assert engine.get_history(f"{mode}/sym_abs_rel_err_max").get_last_value() == 0.0
    assert engine.get_history(f"{mode}/sym_abs_rel_err_min").get_last_value() == 0.0
    assert engine.get_history(f"{mode}/sym_abs_rel_err_sum").get_last_value() == 0.0
    assert engine.get_history(f"{mode}/sym_abs_rel_err_num_predictions").get_last_value() == 4


@mark.parametrize("device", get_available_devices())
def test_symmetric_absolute_relative_error_events_train(device: str, engine: BaseEngine):
    device = torch.device(device)
    metric = SymmetricAbsoluteRelativeError(ct.TRAIN).to(device=device)
    metric.attach(engine)
    metric(torch.ones(2, 2, device=device), 2 * torch.ones(2, 2, device=device))
    engine.fire_event(EngineEvents.TRAIN_EPOCH_STARTED)
    metric(torch.eye(2, device=device) + 1, torch.eye(2, device=device) + 1)
    engine.fire_event(EngineEvents.TRAIN_EPOCH_COMPLETED)
    assert engine.get_history(f"{ct.TRAIN}/sym_abs_rel_err_mean").get_last_value() == 0.0
    assert engine.get_history(f"{ct.TRAIN}/sym_abs_rel_err_max").get_last_value() == 0.0
    assert engine.get_history(f"{ct.TRAIN}/sym_abs_rel_err_min").get_last_value() == 0.0
    assert engine.get_history(f"{ct.TRAIN}/sym_abs_rel_err_sum").get_last_value() == 0.0
    assert engine.get_history(f"{ct.TRAIN}/sym_abs_rel_err_num_predictions").get_last_value() == 4


@mark.parametrize("device", get_available_devices())
def test_symmetric_absolute_relative_error_events_eval(device: str, engine: BaseEngine):
    device = torch.device(device)
    metric = SymmetricAbsoluteRelativeError(ct.EVAL).to(device=device)
    metric.attach(engine)
    metric(torch.ones(2, 2, device=device), 2 * torch.ones(2, 2, device=device))
    engine.fire_event(EngineEvents.EVAL_EPOCH_STARTED)
    metric(torch.eye(2, device=device) + 1, torch.eye(2, device=device) + 1)
    engine.fire_event(EngineEvents.EVAL_EPOCH_COMPLETED)
    assert engine.get_history(f"{ct.EVAL}/sym_abs_rel_err_mean").get_last_value() == 0.0
    assert engine.get_history(f"{ct.EVAL}/sym_abs_rel_err_max").get_last_value() == 0.0
    assert engine.get_history(f"{ct.EVAL}/sym_abs_rel_err_min").get_last_value() == 0.0
    assert engine.get_history(f"{ct.EVAL}/sym_abs_rel_err_sum").get_last_value() == 0.0
    assert engine.get_history(f"{ct.EVAL}/sym_abs_rel_err_num_predictions").get_last_value() == 4


def test_symmetric_absolute_relative_error_reset() -> None:
    state = Mock(spec=BaseState)
    metric = SymmetricAbsoluteRelativeError(ct.EVAL, state=state)
    metric.reset()
    state.reset.assert_called_once_with()
