import torch
from pytest import fixture, mark, raises

from gravitorch import constants as ct
from gravitorch.engines import BaseEngine, EngineEvents
from gravitorch.models.metrics import EmptyMetricError, NormalizedMeanSquaredError
from gravitorch.testing import create_dummy_engine
from gravitorch.utils import get_available_devices
from gravitorch.utils.events import VanillaEventHandler
from gravitorch.utils.history import MinScalarHistory

MODES = (ct.TRAIN, ct.EVAL)
NAMES = ("nmse", "normalized_mean_squared_error")
SIZES = (1, 2)
DTYPES = (torch.long, torch.float)


@fixture
def engine() -> BaseEngine:
    return create_dummy_engine()


################################################
#     Tests for NormalizedMeanSquaredError     #
################################################


@mark.parametrize("mode", MODES)
def test_normalized_mean_squared_error_str(mode: str) -> None:
    assert str(NormalizedMeanSquaredError(mode)).startswith("NormalizedMeanSquaredError(")


def test_normalized_mean_squared_error_init() -> None:
    metric = NormalizedMeanSquaredError(ct.EVAL)
    assert metric._sum_squared_errors == 0.0
    assert metric._sum_squared_targets == 0.0
    assert metric._num_predictions == 0


@mark.parametrize("name", NAMES)
def test_normalized_mean_squared_error_attach_train(name: str, engine: BaseEngine) -> None:
    metric = NormalizedMeanSquaredError(ct.TRAIN, name=name)
    metric.attach(engine)
    assert engine.has_history(f"{ct.TRAIN}/{name}")
    assert isinstance(engine.get_history(f"{ct.TRAIN}/{name}"), MinScalarHistory)
    assert engine.has_event_handler(
        VanillaEventHandler(metric.reset), EngineEvents.TRAIN_EPOCH_STARTED
    )
    assert engine.has_event_handler(
        VanillaEventHandler(metric.value, handler_kwargs={"engine": engine}),
        EngineEvents.TRAIN_EPOCH_COMPLETED,
    )


@mark.parametrize("name", NAMES)
def test_normalized_mean_squared_error_attach_eval(name: str, engine: BaseEngine) -> None:
    metric = NormalizedMeanSquaredError(ct.EVAL, name=name)
    metric.attach(engine)
    assert engine.has_history(f"{ct.EVAL}/{name}")
    assert isinstance(engine.get_history(f"{ct.EVAL}/{name}"), MinScalarHistory)
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
def test_normalized_mean_squared_error_forward_correct(
    device: str, mode: str, batch_size: int, feature_size: int
) -> None:
    device = torch.device(device)
    metric = NormalizedMeanSquaredError(mode).to(device=device)
    metric(
        torch.ones(batch_size, feature_size, device=device),
        torch.ones(batch_size, feature_size, device=device),
    )
    assert metric.value() == {
        f"{mode}/nmse": 0.0,
        f"{mode}/nmse_num_predictions": batch_size * feature_size,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("feature_size", SIZES)
def test_normalized_mean_squared_error_forward_incorrect(
    device: str, mode: str, batch_size: int, feature_size: int
) -> None:
    device = torch.device(device)
    metric = NormalizedMeanSquaredError(mode).to(device=device)
    metric(
        -torch.ones(batch_size, feature_size, device=device),
        torch.ones(batch_size, feature_size, device=device),
    )
    assert metric.value() == {
        f"{mode}/nmse": 4.0,
        f"{mode}/nmse_num_predictions": batch_size * feature_size,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_normalized_mean_squared_error_forward_partially_correct(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = NormalizedMeanSquaredError(mode).to(device=device)
    metric(torch.eye(2, device=device), -torch.eye(2, device=device))
    assert metric.value() == {f"{mode}/nmse": 4.0, f"{mode}/nmse_num_predictions": 4}


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_normalized_mean_squared_error_forward_1d(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = NormalizedMeanSquaredError(mode).to(device=device)
    metric(torch.ones(2, device=device), torch.ones(2, device=device))
    assert metric.value() == {f"{mode}/nmse": 0.0, f"{mode}/nmse_num_predictions": 2}


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_normalized_mean_squared_error_forward_2d(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = NormalizedMeanSquaredError(mode).to(device=device)
    metric(torch.ones(2, 3, device=device), torch.ones(2, 3, device=device))
    assert metric.value() == {f"{mode}/nmse": 0.0, f"{mode}/nmse_num_predictions": 6}


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_normalized_mean_squared_error_forward_3d(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = NormalizedMeanSquaredError(mode).to(device=device)
    metric(torch.ones(2, 3, 4, device=device), torch.ones(2, 3, 4, device=device))
    assert metric.value() == {f"{mode}/nmse": 0.0, f"{mode}/nmse_num_predictions": 24}


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
@mark.parametrize("dtype_prediction", DTYPES)
@mark.parametrize("dtype_target", DTYPES)
def test_normalized_mean_squared_error_forward_dtypes(
    device: str,
    mode: str,
    dtype_prediction: torch.dtype,
    dtype_target: torch.dtype,
) -> None:
    device = torch.device(device)
    metric = NormalizedMeanSquaredError(mode).to(device=device)
    metric(
        torch.ones(2, 3, device=device, dtype=dtype_prediction),
        torch.ones(2, 3, device=device, dtype=dtype_target),
    )
    assert metric.value() == {f"{mode}/nmse": 0.0, f"{mode}/nmse_num_predictions": 6}


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_normalized_mean_squared_error_forward_multiple_batches(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = NormalizedMeanSquaredError(mode).to(device=device)
    metric(torch.ones(2, 2, device=device), torch.ones(2, 2, device=device))
    metric(torch.ones(2, 2, device=device), -torch.ones(2, 2, device=device))
    assert metric.value() == {f"{mode}/nmse": 2.0, f"{mode}/nmse_num_predictions": 8}


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_normalized_mean_squared_error_forward_multiple_batches_with_reset(
    device: str, mode: str
) -> None:
    device = torch.device(device)
    metric = NormalizedMeanSquaredError(mode).to(device=device)
    metric(torch.ones(2, 2, device=device), torch.ones(2, 2, device=device))
    metric.reset()
    metric(torch.eye(2, device=device), -torch.eye(2, device=device))
    assert metric.value() == {f"{mode}/nmse": 4.0, f"{mode}/nmse_num_predictions": 4}


@mark.parametrize("mode", MODES)
def test_normalized_mean_squared_error_value_empty(mode: str) -> None:
    with raises(EmptyMetricError):
        NormalizedMeanSquaredError(mode).value()


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_normalized_mean_squared_error_value_log_engine(
    device: str, mode: str, engine: BaseEngine
) -> None:
    device = torch.device(device)
    metric = NormalizedMeanSquaredError(mode).to(device=device)
    metric(torch.eye(2, device=device), -torch.eye(2, device=device))
    metric.value(engine)
    assert engine.get_history(f"{mode}/nmse").get_last_value() == 4.0
    assert engine.get_history(f"{mode}/nmse_num_predictions").get_last_value() == 4


@mark.parametrize("device", get_available_devices())
def test_normalized_mean_squared_error_events_train(device: str, engine: BaseEngine) -> None:
    device = torch.device(device)
    metric = NormalizedMeanSquaredError(ct.TRAIN).to(device=device)
    metric.attach(engine)
    metric(torch.ones(2, 2, device=device), torch.ones(2, 2, device=device))
    engine.fire_event(EngineEvents.TRAIN_EPOCH_STARTED)
    metric(torch.eye(2, device=device), -torch.eye(2, device=device))
    engine.fire_event(EngineEvents.TRAIN_EPOCH_COMPLETED)
    assert engine.get_history(f"{ct.TRAIN}/nmse").get_last_value() == 4.0
    assert engine.get_history(f"{ct.TRAIN}/nmse_num_predictions").get_last_value() == 4


@mark.parametrize("device", get_available_devices())
def test_normalized_mean_squared_error_events_eval(device: str, engine: BaseEngine) -> None:
    device = torch.device(device)
    metric = NormalizedMeanSquaredError(ct.EVAL).to(device=device)
    metric.attach(engine)
    metric(torch.ones(2, 2, device=device), torch.ones(2, 2, device=device))
    engine.fire_event(EngineEvents.EVAL_EPOCH_STARTED)
    metric(torch.eye(2, device=device), -torch.eye(2, device=device))
    engine.fire_event(EngineEvents.EVAL_EPOCH_COMPLETED)
    assert engine.get_history(f"{ct.EVAL}/nmse").get_last_value() == 4.0
    assert engine.get_history(f"{ct.EVAL}/nmse_num_predictions").get_last_value() == 4


def test_normalized_mean_squared_error_reset() -> None:
    metric = NormalizedMeanSquaredError(ct.EVAL)
    metric._sum_squared_errors = 10.0
    metric._sum_squared_targets = 15.0
    metric._num_predictions = 5
    metric.reset()
    assert metric._sum_squared_errors == 0.0
    assert metric._sum_squared_targets == 0.0
    assert metric._num_predictions == 0
