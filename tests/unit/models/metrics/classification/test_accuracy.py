from collections.abc import Sequence
from unittest.mock import Mock

import torch
from objectory import OBJECT_TARGET
from pytest import fixture, mark, raises
from torch.nn import Identity

from gravitorch import constants as ct
from gravitorch.engines import BaseEngine, EngineEvents
from gravitorch.models.metrics import (
    BinaryAccuracy,
    CategoricalAccuracy,
    EmptyMetricError,
    TopKAccuracy,
)
from gravitorch.models.metrics.state import (
    AccuracyState,
    BaseState,
    ExtendedAccuracyState,
)
from gravitorch.nn import ToBinaryLabel
from gravitorch.testing import create_dummy_engine
from gravitorch.utils import get_available_devices
from gravitorch.utils.events import VanillaEventHandler
from gravitorch.utils.history import MaxScalarHistory, MinScalarHistory

MODES = (ct.TRAIN, ct.EVAL)
NAMES = ("name1", "name2")
SIZES = (1, 2)
DTYPES = (torch.long, torch.float)


@fixture
def engine() -> BaseEngine:
    return create_dummy_engine()


####################################
#     Tests for BinaryAccuracy     #
####################################


@mark.parametrize("mode", MODES)
def test_binary_accuracy_str(mode: str) -> None:
    assert str(BinaryAccuracy(mode)).startswith("BinaryAccuracy(")


@mark.parametrize("threshold", (0.0, 0.5))
def test_binary_accuracy_threshold(threshold: float) -> None:
    metric = BinaryAccuracy(ct.EVAL, threshold=threshold)
    assert isinstance(metric.prediction_transform, ToBinaryLabel)
    assert metric.prediction_transform.threshold == threshold


def test_binary_accuracy_threshold_default() -> None:
    assert isinstance(BinaryAccuracy(ct.EVAL).prediction_transform, Identity)


def test_binary_accuracy_state_default() -> None:
    assert isinstance(BinaryAccuracy(ct.EVAL)._state, AccuracyState)


def test_binary_accuracy_state_extended() -> None:
    assert isinstance(
        BinaryAccuracy(ct.EVAL, state=ExtendedAccuracyState())._state, ExtendedAccuracyState
    )


@mark.parametrize("name", NAMES)
def test_binary_accuracy_attach_train(name: str, engine: BaseEngine) -> None:
    metric = BinaryAccuracy(ct.TRAIN, name=name)
    metric.attach(engine)
    assert isinstance(engine.get_history(f"{ct.TRAIN}/{name}_accuracy"), MaxScalarHistory)
    assert engine.has_event_handler(
        VanillaEventHandler(metric.reset), EngineEvents.TRAIN_EPOCH_STARTED
    )
    assert engine.has_event_handler(
        VanillaEventHandler(metric.value, handler_kwargs={"engine": engine}),
        EngineEvents.TRAIN_EPOCH_COMPLETED,
    )


@mark.parametrize("name", NAMES)
def test_binary_accuracy_attach_eval(name: str, engine: BaseEngine) -> None:
    metric = BinaryAccuracy(ct.EVAL, name=name)
    metric.attach(engine)
    assert isinstance(engine.get_history(f"{ct.EVAL}/{name}_accuracy"), MaxScalarHistory)
    assert engine.has_event_handler(
        VanillaEventHandler(metric.reset), EngineEvents.EVAL_EPOCH_STARTED
    )
    assert engine.has_event_handler(
        VanillaEventHandler(metric.value, handler_kwargs={"engine": engine}),
        EngineEvents.EVAL_EPOCH_COMPLETED,
    )


def test_binary_accuracy_attach_state_extended(engine: BaseEngine) -> None:
    metric = BinaryAccuracy(ct.EVAL, state=ExtendedAccuracyState())
    metric.attach(engine)
    assert isinstance(engine.get_history(f"{ct.EVAL}/bin_acc_accuracy"), MaxScalarHistory)
    assert isinstance(engine.get_history(f"{ct.EVAL}/bin_acc_error"), MinScalarHistory)
    assert isinstance(
        engine.get_history(f"{ct.EVAL}/bin_acc_num_correct_predictions"),
        MaxScalarHistory,
    )
    assert isinstance(
        engine.get_history(f"{ct.EVAL}/bin_acc_num_incorrect_predictions"),
        MinScalarHistory,
    )
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
def test_binary_accuracy_forward_correct(device: str, mode: str, batch_size: int) -> None:
    device = torch.device(device)
    metric = BinaryAccuracy(mode).to(device=device)
    metric(torch.ones(batch_size, device=device), torch.ones(batch_size, device=device))
    assert metric.value() == {
        f"{mode}/bin_acc_accuracy": 1.0,
        f"{mode}/bin_acc_num_predictions": batch_size,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
@mark.parametrize("batch_size", SIZES)
def test_binary_accuracy_forward_incorrect(device: str, mode: str, batch_size: int) -> None:
    device = torch.device(device)
    metric = BinaryAccuracy(mode).to(device=device)
    metric(torch.zeros(batch_size, device=device), torch.ones(batch_size, device=device))
    assert metric.value() == {
        f"{mode}/bin_acc_accuracy": 0.0,
        f"{mode}/bin_acc_num_predictions": batch_size,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_binary_accuracy_forward_partially_correct(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = BinaryAccuracy(mode).to(device=device)
    metric(torch.tensor([0, 1, 0, 1], device=device), torch.tensor([1, 1, 0, 0], device=device))
    assert metric.value() == {
        f"{mode}/bin_acc_accuracy": 0.5,
        f"{mode}/bin_acc_num_predictions": 4,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_binary_accuracy_forward_1d(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = BinaryAccuracy(mode).to(device=device)
    metric(torch.ones(2, device=device), torch.ones(2, device=device))
    assert metric.value() == {
        f"{mode}/bin_acc_accuracy": 1.0,
        f"{mode}/bin_acc_num_predictions": 2,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_binary_accuracy_forward_1d_and_2d(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = BinaryAccuracy(mode).to(device=device)
    metric(torch.ones(2, device=device), torch.ones(2, 1, device=device))
    assert metric.value() == {
        f"{mode}/bin_acc_accuracy": 1.0,
        f"{mode}/bin_acc_num_predictions": 2,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_binary_accuracy_forward_2d_and_1d(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = BinaryAccuracy(mode).to(device=device)
    metric(torch.ones(2, 1, device=device), torch.ones(2, device=device))
    assert metric.value() == {
        f"{mode}/bin_acc_accuracy": 1.0,
        f"{mode}/bin_acc_num_predictions": 2,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_binary_accuracy_forward_2d(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = BinaryAccuracy(mode).to(device=device)
    metric(torch.ones(2, 3, device=device), torch.ones(2, 3, device=device))
    assert metric.value() == {
        f"{mode}/bin_acc_accuracy": 1.0,
        f"{mode}/bin_acc_num_predictions": 6,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_binary_accuracy_forward_3d(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = BinaryAccuracy(mode).to(device=device)
    metric(torch.ones(2, 3, 4, device=device), torch.ones(2, 3, 4, device=device))
    assert metric.value() == {
        f"{mode}/bin_acc_accuracy": 1.0,
        f"{mode}/bin_acc_num_predictions": 24,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
@mark.parametrize("dtype_prediction", (torch.bool, torch.long, torch.float))
@mark.parametrize("dtype_target", (torch.bool, torch.long, torch.float))
def test_binary_accuracy_forward_dtypes(
    device: str,
    mode: str,
    dtype_prediction: torch.dtype,
    dtype_target: torch.dtype,
) -> None:
    device = torch.device(device)
    metric = BinaryAccuracy(mode).to(device=device)
    metric(
        torch.ones(2, 3, device=device, dtype=dtype_prediction),
        torch.ones(2, 3, device=device, dtype=dtype_target),
    )
    assert metric.value() == {
        f"{mode}/bin_acc_accuracy": 1.0,
        f"{mode}/bin_acc_num_predictions": 6,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_binary_accuracy_forward_state(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = BinaryAccuracy(mode, state=ExtendedAccuracyState()).to(device=device)
    metric(torch.ones(2, 3, device=device), torch.ones(2, 3, device=device))
    assert metric.value() == {
        f"{mode}/bin_acc_accuracy": 1.0,
        f"{mode}/bin_acc_error": 0.0,
        f"{mode}/bin_acc_num_correct_predictions": 6,
        f"{mode}/bin_acc_num_incorrect_predictions": 0,
        f"{mode}/bin_acc_num_predictions": 6,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_binary_accuracy_forward_threshold_0(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = BinaryAccuracy(mode, threshold=0.0).to(device=device)
    metric(torch.tensor([-1, 1, -2, 1], device=device), torch.tensor([1, 1, 0, 0], device=device))
    assert metric.value() == {
        f"{mode}/bin_acc_accuracy": 0.5,
        f"{mode}/bin_acc_num_predictions": 4,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_binary_accuracy_forward_threshold_0_5(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = BinaryAccuracy(mode, threshold=0.5).to(device=device)
    metric(
        torch.tensor([0.1, 0.6, 0.4, 1.0], device=device), torch.tensor([1, 1, 0, 0], device=device)
    )
    assert metric.value() == {
        f"{mode}/bin_acc_accuracy": 0.5,
        f"{mode}/bin_acc_num_predictions": 4,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_binary_accuracy_forward_multiple_batches(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = BinaryAccuracy(mode).to(device=device)
    metric(torch.zeros(4, device=device), torch.ones(4, device=device))
    metric(torch.ones(4, device=device), torch.ones(4, device=device))
    assert metric.value() == {
        f"{mode}/bin_acc_accuracy": 0.5,
        f"{mode}/bin_acc_num_predictions": 8,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_binary_accuracy_forward_multiple_batches_with_reset(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = BinaryAccuracy(mode).to(device=device)
    metric(torch.zeros(4, device=device), torch.ones(4, device=device))
    metric.reset()
    metric(torch.ones(4, device=device), torch.ones(4, device=device))
    assert metric.value() == {
        f"{mode}/bin_acc_accuracy": 1.0,
        f"{mode}/bin_acc_num_predictions": 4,
    }


@mark.parametrize("mode", MODES)
def test_binary_accuracy_value_empty(mode: bool) -> None:
    with raises(EmptyMetricError):
        BinaryAccuracy(mode).value()


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_binary_accuracy_value_log_engine(device: str, mode: str, engine: BaseEngine) -> None:
    device = torch.device(device)
    metric = BinaryAccuracy(mode).to(device=device)
    metric(torch.ones(4, device=device), torch.ones(4, device=device))
    metric.value(engine)
    assert engine.get_history(f"{mode}/bin_acc_accuracy").get_last_value() == 1.0
    assert engine.get_history(f"{mode}/bin_acc_num_predictions").get_last_value() == 4


@mark.parametrize("device", get_available_devices())
def test_binary_accuracy_events_train(device: str, engine: BaseEngine) -> None:
    device = torch.device(device)
    metric = BinaryAccuracy(ct.TRAIN).to(device=device)
    metric.attach(engine)
    metric(torch.zeros(4, device=device), torch.ones(4, device=device))
    engine.fire_event(EngineEvents.TRAIN_EPOCH_STARTED)
    metric(torch.ones(4, device=device), torch.ones(4, device=device))
    engine.fire_event(EngineEvents.TRAIN_EPOCH_COMPLETED)
    assert engine.get_history(f"{ct.TRAIN}/bin_acc_accuracy").get_last_value() == 1.0
    assert engine.get_history(f"{ct.TRAIN}/bin_acc_num_predictions").get_last_value() == 4


@mark.parametrize("device", get_available_devices())
def test_binary_accuracy_events_eval(device: str, engine: BaseEngine) -> None:
    device = torch.device(device)
    metric = BinaryAccuracy(ct.EVAL).to(device=device)
    metric.attach(engine)
    metric(torch.zeros(4, device=device), torch.ones(4, device=device))
    engine.fire_event(EngineEvents.EVAL_EPOCH_STARTED)
    metric(torch.ones(4, device=device), torch.ones(4, device=device))
    engine.fire_event(EngineEvents.EVAL_EPOCH_COMPLETED)
    assert engine.get_history(f"{ct.EVAL}/bin_acc_accuracy").get_last_value() == 1.0
    assert engine.get_history(f"{ct.EVAL}/bin_acc_num_predictions").get_last_value() == 4


def test_binary_accuracy_reset() -> None:
    state = Mock(spec=BaseState)
    metric = BinaryAccuracy(ct.EVAL, state=state)
    metric.reset()
    state.reset.assert_called_once_with()


#########################################
#     Tests for CategoricalAccuracy     #
#########################################


@mark.parametrize("mode", MODES)
def test_categorical_accuracy_str(mode: str) -> None:
    assert str(CategoricalAccuracy(mode)).startswith("CategoricalAccuracy(")


def test_categorical_accuracy_state_default() -> None:
    assert isinstance(CategoricalAccuracy(ct.EVAL)._state, AccuracyState)


def test_categorical_accuracy_state_extended() -> None:
    assert isinstance(
        CategoricalAccuracy(ct.EVAL, state=ExtendedAccuracyState())._state,
        ExtendedAccuracyState,
    )


@mark.parametrize("name", NAMES)
def test_categorical_accuracy_attach_train(name: str, engine: BaseEngine) -> None:
    metric = CategoricalAccuracy(ct.TRAIN, name=name)
    metric.attach(engine)
    assert isinstance(engine.get_history(f"{ct.TRAIN}/{name}_accuracy"), MaxScalarHistory)
    assert engine.has_event_handler(
        VanillaEventHandler(metric.reset), EngineEvents.TRAIN_EPOCH_STARTED
    )
    assert engine.has_event_handler(
        VanillaEventHandler(metric.value, handler_kwargs={"engine": engine}),
        EngineEvents.TRAIN_EPOCH_COMPLETED,
    )


@mark.parametrize("name", NAMES)
def test_categorical_accuracy_attach_eval(name: str, engine: BaseEngine) -> None:
    metric = CategoricalAccuracy(ct.EVAL, name=name)
    metric.attach(engine)
    assert isinstance(engine.get_history(f"{ct.EVAL}/{name}_accuracy"), MaxScalarHistory)
    assert engine.has_event_handler(
        VanillaEventHandler(metric.reset), EngineEvents.EVAL_EPOCH_STARTED
    )
    assert engine.has_event_handler(
        VanillaEventHandler(metric.value, handler_kwargs={"engine": engine}),
        EngineEvents.EVAL_EPOCH_COMPLETED,
    )


def test_categorical_accuracy_attach_state_extended(engine: BaseEngine) -> None:
    metric = CategoricalAccuracy(ct.EVAL, state=ExtendedAccuracyState())
    metric.attach(engine)
    assert isinstance(engine.get_history(f"{ct.EVAL}/cat_acc_accuracy"), MaxScalarHistory)
    assert isinstance(engine.get_history(f"{ct.EVAL}/cat_acc_error"), MinScalarHistory)
    assert isinstance(
        engine.get_history(f"{ct.EVAL}/cat_acc_num_correct_predictions"),
        MaxScalarHistory,
    )
    assert isinstance(
        engine.get_history(f"{ct.EVAL}/cat_acc_num_incorrect_predictions"),
        MinScalarHistory,
    )
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
def test_categorical_accuracy_forward_correct(device: str, mode: str, batch_size: int) -> None:
    device = torch.device(device)
    metric = CategoricalAccuracy(mode).to(device=device)
    metric(torch.eye(batch_size, device=device), torch.arange(batch_size, device=device))
    assert metric.value() == {
        f"{mode}/cat_acc_accuracy": 1.0,
        f"{mode}/cat_acc_num_predictions": batch_size,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
@mark.parametrize("batch_size", SIZES)
def test_categorical_accuracy_forward_incorrect(device: str, mode: str, batch_size: int) -> None:
    device = torch.device(device)
    metric = CategoricalAccuracy(mode).to(device=device)
    prediction = torch.zeros(batch_size, 3, device=device)
    prediction[:, 0] = 1
    metric(prediction, torch.ones(batch_size, device=device))
    assert metric.value() == {
        f"{mode}/cat_acc_accuracy": 0.0,
        f"{mode}/cat_acc_num_predictions": batch_size,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_categorical_accuracy_forward_partially_correct(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = CategoricalAccuracy(mode).to(device=device)
    metric(torch.eye(2, device=device), torch.zeros(2, device=device))
    assert metric.value() == {
        f"{mode}/cat_acc_accuracy": 0.5,
        f"{mode}/cat_acc_num_predictions": 2,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_categorical_accuracy_forward_prediction_2d(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = CategoricalAccuracy(mode).to(device=device)
    prediction = torch.rand(2, 3, device=device)
    prediction[..., 0] = 2
    metric(prediction, torch.zeros(2, device=device))
    assert metric.value() == {
        f"{mode}/cat_acc_accuracy": 1.0,
        f"{mode}/cat_acc_num_predictions": 2,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_categorical_accuracy_forward_prediction_2d_target_2d(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = CategoricalAccuracy(mode).to(device=device)
    prediction = torch.rand(2, 3, device=device)
    prediction[..., 0] = 2
    metric(prediction, torch.zeros(2, 1, device=device))
    assert metric.value() == {
        f"{mode}/cat_acc_accuracy": 1.0,
        f"{mode}/cat_acc_num_predictions": 2,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_categorical_accuracy_forward_prediction_3d(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = CategoricalAccuracy(mode).to(device=device)
    prediction = torch.rand(2, 3, 4, device=device)
    prediction[..., 0] = 2
    metric(prediction, torch.zeros(2, 3, device=device))
    assert metric.value() == {
        f"{mode}/cat_acc_accuracy": 1.0,
        f"{mode}/cat_acc_num_predictions": 6,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_categorical_accuracy_forward_prediction_4d(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = CategoricalAccuracy(mode).to(device=device)
    prediction = torch.rand(2, 3, 4, 5, device=device)
    prediction[..., 0] = 2
    metric(prediction, torch.zeros(2, 3, 4, device=device))
    assert metric.value() == {
        f"{mode}/cat_acc_accuracy": 1.0,
        f"{mode}/cat_acc_num_predictions": 24,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
@mark.parametrize("dtype_prediction", DTYPES)
@mark.parametrize("dtype_target", DTYPES)
def test_categorical_accuracy_forward_dtypes(
    device: str,
    mode: str,
    dtype_prediction: torch.dtype,
    dtype_target: torch.dtype,
) -> None:
    device = torch.device(device)
    metric = CategoricalAccuracy(mode).to(device=device)
    metric(
        torch.eye(2, device=device, dtype=dtype_prediction),
        torch.arange(2, device=device, dtype=dtype_target),
    )
    assert metric.value() == {
        f"{mode}/cat_acc_accuracy": 1.0,
        f"{mode}/cat_acc_num_predictions": 2,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_categorical_accuracy_forward_state(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = CategoricalAccuracy(mode, state=ExtendedAccuracyState()).to(device=device)
    metric(torch.eye(2, device=device), torch.zeros(2, device=device))
    assert metric.value() == {
        f"{mode}/cat_acc_accuracy": 0.5,
        f"{mode}/cat_acc_error": 0.5,
        f"{mode}/cat_acc_num_correct_predictions": 1,
        f"{mode}/cat_acc_num_incorrect_predictions": 1,
        f"{mode}/cat_acc_num_predictions": 2,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_categorical_accuracy_forward_multiple_batches(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = CategoricalAccuracy(mode).to(device=device)
    metric(torch.eye(2, device=device), torch.zeros(2, device=device))
    metric(torch.eye(2, device=device), torch.tensor([0, 1], device=device))
    assert metric.value() == {
        f"{mode}/cat_acc_accuracy": 0.75,
        f"{mode}/cat_acc_num_predictions": 4,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_categorical_accuracy_forward_multiple_batches_with_reset(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = CategoricalAccuracy(mode).to(device=device)
    metric(torch.eye(2, device=device), torch.zeros(2, device=device))
    metric.reset()
    metric(torch.eye(2, device=device), torch.tensor([0, 1], device=device))
    assert metric.value() == {
        f"{mode}/cat_acc_accuracy": 1.0,
        f"{mode}/cat_acc_num_predictions": 2,
    }


@mark.parametrize("mode", MODES)
def test_categorical_accuracy_value_empty(mode: str) -> None:
    with raises(EmptyMetricError):
        CategoricalAccuracy(mode).value()


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_categorical_accuracy_value_log_engine(device: str, mode: str, engine: BaseEngine) -> None:
    device = torch.device(device)
    metric = CategoricalAccuracy(mode).to(device=device)
    metric(torch.eye(2, device=device), torch.zeros(2, device=device))
    metric.value(engine)
    assert engine.get_history(f"{mode}/cat_acc_accuracy").get_last_value() == 0.5
    assert engine.get_history(f"{mode}/cat_acc_num_predictions").get_last_value() == 2


@mark.parametrize("device", get_available_devices())
def test_categorical_accuracy_events_train(device: str, engine: BaseEngine) -> None:
    device = torch.device(device)
    metric = CategoricalAccuracy(ct.TRAIN).to(device=device)
    metric.attach(engine)
    metric(torch.eye(2, device=device), torch.zeros(2, device=device))
    engine.fire_event(EngineEvents.TRAIN_EPOCH_STARTED)
    metric(torch.eye(2, device=device), torch.tensor([0, 1], device=device))
    engine.fire_event(EngineEvents.TRAIN_EPOCH_COMPLETED)
    assert engine.get_history(f"{ct.TRAIN}/cat_acc_accuracy").get_last_value() == 1.0
    assert engine.get_history(f"{ct.TRAIN}/cat_acc_num_predictions").get_last_value() == 2


@mark.parametrize("device", get_available_devices())
def test_categorical_accuracy_events_eval(device: str, engine: BaseEngine) -> None:
    device = torch.device(device)
    metric = CategoricalAccuracy(ct.EVAL).to(device=device)
    metric.attach(engine)
    metric(torch.eye(2, device=device), torch.zeros(2, device=device))
    engine.fire_event(EngineEvents.EVAL_EPOCH_STARTED)
    metric(torch.eye(2, device=device), torch.tensor([0, 1], device=device))
    engine.fire_event(EngineEvents.EVAL_EPOCH_COMPLETED)
    assert engine.get_history(f"{ct.EVAL}/cat_acc_accuracy").get_last_value() == 1.0
    assert engine.get_history(f"{ct.EVAL}/cat_acc_num_predictions").get_last_value() == 2


def test_categorical_accuracy_reset() -> None:
    state = Mock(spec=BaseState)
    metric = CategoricalAccuracy(ct.EVAL, state=state)
    metric.reset()
    state.reset.assert_called_once_with()


##################################
#     Tests for TopKAccuracy     #
##################################


@mark.parametrize("mode", MODES)
def test_top_k_accuracy_str(mode: str) -> None:
    assert str(TopKAccuracy(mode)).startswith("TopKAccuracy(")


@mark.parametrize(
    "topk,tuple_topk",
    (
        ((1,), (1,)),
        ((1, 5), (1, 5)),
        ([1], (1,)),
        ([1, 5], (1, 5)),
    ),
)
def test_top_k_accuracy_tolerances(topk: Sequence[int], tuple_topk: tuple[int, ...]) -> None:
    assert TopKAccuracy(mode=ct.TRAIN, topk=topk).topk == tuple_topk


def test_top_k_accuracy_state_config_default() -> None:
    metric = TopKAccuracy(ct.EVAL, topk=(1, 5))
    assert len(metric._states) == 2
    assert isinstance(metric._states[1], AccuracyState)
    assert metric._states[1].num_predictions == 0
    assert isinstance(metric._states[5], AccuracyState)
    assert metric._states[5].num_predictions == 0


def test_top_k_accuracy_state_config_extended() -> None:
    metric = TopKAccuracy(
        ct.EVAL,
        topk=(1, 5),
        state_config={OBJECT_TARGET: "gravitorch.models.metrics.state.ExtendedAccuracyState"},
    )
    assert len(metric._states) == 2
    assert isinstance(metric._states[1], ExtendedAccuracyState)
    assert metric._states[1].num_predictions == 0
    assert isinstance(metric._states[5], ExtendedAccuracyState)
    assert metric._states[5].num_predictions == 0


@mark.parametrize("name", NAMES)
def test_top_k_accuracy_attach_train(name: str, engine: BaseEngine) -> None:
    metric = TopKAccuracy(ct.TRAIN, name=name, topk=(1, 5))
    metric.attach(engine)
    assert isinstance(engine.get_history(f"{ct.TRAIN}/{name}_1_accuracy"), MaxScalarHistory)
    assert isinstance(engine.get_history(f"{ct.TRAIN}/{name}_5_accuracy"), MaxScalarHistory)
    assert engine.has_event_handler(
        VanillaEventHandler(metric.reset), EngineEvents.TRAIN_EPOCH_STARTED
    )
    assert engine.has_event_handler(
        VanillaEventHandler(metric.value, handler_kwargs={"engine": engine}),
        EngineEvents.TRAIN_EPOCH_COMPLETED,
    )


@mark.parametrize("name", NAMES)
def test_top_k_accuracy_attach_eval(name: str, engine: BaseEngine) -> None:
    metric = TopKAccuracy(ct.EVAL, name=name, topk=(1, 5))
    metric.attach(engine)
    assert isinstance(engine.get_history(f"{ct.EVAL}/{name}_1_accuracy"), MaxScalarHistory)
    assert isinstance(engine.get_history(f"{ct.EVAL}/{name}_5_accuracy"), MaxScalarHistory)
    assert engine.has_event_handler(
        VanillaEventHandler(metric.reset), EngineEvents.EVAL_EPOCH_STARTED
    )
    assert engine.has_event_handler(
        VanillaEventHandler(metric.value, handler_kwargs={"engine": engine}),
        EngineEvents.EVAL_EPOCH_COMPLETED,
    )


def test_top_k_accuracy_attach_state_extended(engine: BaseEngine) -> None:
    metric = TopKAccuracy(
        ct.EVAL,
        topk=(1, 5),
        state_config={OBJECT_TARGET: "gravitorch.models.metrics.state.ExtendedAccuracyState"},
    )
    metric.attach(engine)
    assert isinstance(engine.get_history(f"{ct.EVAL}/acc_top_1_accuracy"), MaxScalarHistory)
    assert isinstance(engine.get_history(f"{ct.EVAL}/acc_top_1_error"), MinScalarHistory)
    assert isinstance(
        engine.get_history(f"{ct.EVAL}/acc_top_1_num_correct_predictions"), MaxScalarHistory
    )
    assert isinstance(
        engine.get_history(f"{ct.EVAL}/acc_top_1_num_incorrect_predictions"),
        MinScalarHistory,
    )
    assert isinstance(engine.get_history(f"{ct.EVAL}/acc_top_5_accuracy"), MaxScalarHistory)
    assert isinstance(engine.get_history(f"{ct.EVAL}/acc_top_5_error"), MinScalarHistory)
    assert isinstance(
        engine.get_history(f"{ct.EVAL}/acc_top_5_num_correct_predictions"), MaxScalarHistory
    )
    assert isinstance(
        engine.get_history(f"{ct.EVAL}/acc_top_5_num_incorrect_predictions"),
        MinScalarHistory,
    )
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
def test_top_k_accuracy_forward_top1_correct(device: str, mode: str, batch_size: int) -> None:
    device = torch.device(device)
    metric = TopKAccuracy(mode, topk=(1,)).to(device=device)
    metric(
        prediction=torch.eye(batch_size, device=device),
        target=torch.arange(batch_size, device=device),
    )
    assert metric.value() == {
        f"{mode}/acc_top_1_accuracy": 1.0,
        f"{mode}/acc_top_1_num_predictions": batch_size,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_top_k_accuracy_forward_top_1_5_correct(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = TopKAccuracy(mode, topk=(1, 5)).to(device=device)
    metric(prediction=torch.eye(10, device=device), target=torch.arange(10, device=device))
    assert metric.value() == {
        f"{mode}/acc_top_1_accuracy": 1.0,
        f"{mode}/acc_top_1_num_predictions": 10,
        f"{mode}/acc_top_5_accuracy": 1.0,
        f"{mode}/acc_top_5_num_predictions": 10,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_top_k_accuracy_forward_top_1_partially_correct(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = TopKAccuracy(mode, topk=(1,)).to(device=device)
    metric(
        prediction=torch.tensor([[0, 2, 1], [2, 1, 0]], device=device),
        target=torch.tensor([1, 2], device=device),
    )
    assert metric.value() == {
        f"{mode}/acc_top_1_accuracy": 0.5,
        f"{mode}/acc_top_1_num_predictions": 2,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_top_k_accuracy_forward_top_1_2_3_partially_correct(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = TopKAccuracy(mode, topk=(1, 2, 3)).to(device=device)
    metric(
        prediction=torch.tensor([[0, 2, 1], [2, 1, 0]], device=device),
        target=torch.tensor([1, 2], device=device),
    )
    assert metric.value() == {
        f"{mode}/acc_top_1_accuracy": 0.5,
        f"{mode}/acc_top_1_num_predictions": 2,
        f"{mode}/acc_top_2_accuracy": 0.5,
        f"{mode}/acc_top_2_num_predictions": 2,
        f"{mode}/acc_top_3_accuracy": 1.0,
        f"{mode}/acc_top_3_num_predictions": 2,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_top_k_accuracy_forward_top_1_incorrect(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = TopKAccuracy(mode, topk=(1,)).to(device=device)
    metric(
        prediction=torch.tensor([[0, 1, 2], [2, 1, 0]], device=device),
        target=torch.tensor([1, 2], device=device),
    )
    assert metric.value() == {
        f"{mode}/acc_top_1_accuracy": 0.0,
        f"{mode}/acc_top_1_num_predictions": 2,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_top_k_accuracy_forward_top_1_2_3_incorrect(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = TopKAccuracy(mode, topk=(1, 2, 3)).to(device=device)
    metric(
        prediction=torch.tensor([[0, 1, 2], [2, 1, 0]], device=device),
        target=torch.tensor([1, 2], device=device),
    )
    assert metric.value() == {
        f"{mode}/acc_top_1_accuracy": 0.0,
        f"{mode}/acc_top_1_num_predictions": 2,
        f"{mode}/acc_top_2_accuracy": 0.5,
        f"{mode}/acc_top_2_num_predictions": 2,
        f"{mode}/acc_top_3_accuracy": 1.0,
        f"{mode}/acc_top_3_num_predictions": 2,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_top_k_accuracy_forward_prediction_2d(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = TopKAccuracy(mode, topk=(1,)).to(device=device)
    prediction = torch.rand(2, 3, device=device)
    prediction[..., 0] = 2
    metric(prediction=prediction, target=torch.zeros(2, device=device))
    assert metric.value() == {
        f"{mode}/acc_top_1_accuracy": 1.0,
        f"{mode}/acc_top_1_num_predictions": 2,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_top_k_accuracy_forward_prediction_2d_target_2d(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = TopKAccuracy(mode, topk=(1,)).to(device=device)
    prediction = torch.rand(2, 3, device=device)
    prediction[..., 0] = 2
    metric(prediction=prediction, target=torch.zeros(2, 1, device=device))
    assert metric.value() == {
        f"{mode}/acc_top_1_accuracy": 1.0,
        f"{mode}/acc_top_1_num_predictions": 2,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_top_k_accuracy_forward_prediction_3d(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = TopKAccuracy(mode, topk=(1,)).to(device=device)
    prediction = torch.rand(2, 3, 4, device=device)
    prediction[..., 0] = 2
    metric(prediction=prediction, target=torch.zeros(2, 3, device=device))
    assert metric.value() == {
        f"{mode}/acc_top_1_accuracy": 1.0,
        f"{mode}/acc_top_1_num_predictions": 6,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_top_k_accuracy_forward_prediction_4d(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = TopKAccuracy(mode, topk=(1,)).to(device=device)
    prediction = torch.rand(2, 3, 4, 5, device=device)
    prediction[..., 0] = 2
    metric(prediction=prediction, target=torch.zeros(2, 3, 4, device=device))
    assert metric.value() == {
        f"{mode}/acc_top_1_accuracy": 1.0,
        f"{mode}/acc_top_1_num_predictions": 24,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
@mark.parametrize("dtype_prediction", DTYPES)
@mark.parametrize("dtype_target", DTYPES)
def test_top_k_accuracy_forward_dtypes(
    device: str,
    mode: str,
    dtype_prediction: torch.dtype,
    dtype_target: torch.dtype,
) -> None:
    device = torch.device(device)
    metric = TopKAccuracy(mode, topk=(1,)).to(device=device)
    metric(
        torch.eye(4, device=device, dtype=dtype_prediction),
        torch.arange(4, device=device, dtype=dtype_target),
    )
    assert metric.value() == {
        f"{mode}/acc_top_1_accuracy": 1.0,
        f"{mode}/acc_top_1_num_predictions": 4,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_top_k_accuracy_forward_state(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = TopKAccuracy(
        mode,
        topk=(1, 2, 3),
        state_config={OBJECT_TARGET: "gravitorch.models.metrics.state.ExtendedAccuracyState"},
    ).to(device=device)
    metric(
        prediction=torch.tensor([[0, 2, 1], [2, 1, 0]], device=device),
        target=torch.tensor([1, 2], device=device),
    )
    assert metric.value() == {
        f"{mode}/acc_top_1_accuracy": 0.5,
        f"{mode}/acc_top_1_error": 0.5,
        f"{mode}/acc_top_1_num_correct_predictions": 1,
        f"{mode}/acc_top_1_num_incorrect_predictions": 1,
        f"{mode}/acc_top_1_num_predictions": 2,
        f"{mode}/acc_top_2_accuracy": 0.5,
        f"{mode}/acc_top_2_error": 0.5,
        f"{mode}/acc_top_2_num_correct_predictions": 1,
        f"{mode}/acc_top_2_num_incorrect_predictions": 1,
        f"{mode}/acc_top_2_num_predictions": 2,
        f"{mode}/acc_top_3_accuracy": 1.0,
        f"{mode}/acc_top_3_error": 0.0,
        f"{mode}/acc_top_3_num_correct_predictions": 2,
        f"{mode}/acc_top_3_num_incorrect_predictions": 0,
        f"{mode}/acc_top_3_num_predictions": 2,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_top_k_accuracy_forward_top_1_multiple_batches(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = TopKAccuracy(mode, topk=(1,)).to(device=device)
    metric(
        prediction=torch.tensor([[0, 2, 1], [2, 1, 0]], device=device),
        target=torch.tensor([1, 0], device=device),
    )
    metric(
        prediction=torch.tensor([[0, 2, 1], [2, 1, 0]], device=device),
        target=torch.tensor([1, 2], device=device),
    )
    assert metric.value() == {
        f"{mode}/acc_top_1_accuracy": 0.75,
        f"{mode}/acc_top_1_num_predictions": 4,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_top_k_accuracy_forward_top_1_multiple_batches_with_reset(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = TopKAccuracy(mode, topk=(1,)).to(device=device)
    metric(
        prediction=torch.tensor([[0, 2, 1], [2, 1, 0]], device=device),
        target=torch.tensor([1, 0], device=device),
    )
    metric.reset()
    metric(
        prediction=torch.tensor([[0, 2, 1], [2, 1, 0]], device=device),
        target=torch.tensor([1, 2], device=device),
    )
    assert metric.value() == {
        f"{mode}/acc_top_1_accuracy": 0.5,
        f"{mode}/acc_top_1_num_predictions": 2,
    }


@mark.parametrize("mode", MODES)
def test_top_k_accuracy_value_empty(mode: str) -> None:
    with raises(EmptyMetricError):
        TopKAccuracy(mode).value()


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_top_k_accuracy_value_log_engine(device: str, mode: str, engine: BaseEngine) -> None:
    device = torch.device(device)
    metric = TopKAccuracy(mode, topk=(1, 2, 3)).to(device=device)
    metric(
        prediction=torch.tensor([[0, 2, 1], [2, 1, 0]], device=device),
        target=torch.tensor([1, 2], device=device),
    )
    metric.value(engine)
    assert engine.get_history(f"{mode}/acc_top_1_accuracy").get_last_value() == 0.5
    assert engine.get_history(f"{mode}/acc_top_1_num_predictions").get_last_value() == 2
    assert engine.get_history(f"{mode}/acc_top_2_accuracy").get_last_value() == 0.5
    assert engine.get_history(f"{mode}/acc_top_2_num_predictions").get_last_value() == 2
    assert engine.get_history(f"{mode}/acc_top_3_accuracy").get_last_value() == 1.0
    assert engine.get_history(f"{mode}/acc_top_3_num_predictions").get_last_value() == 2


@mark.parametrize("device", get_available_devices())
def test_top_k_accuracy_events_train(device: str, engine: BaseEngine) -> None:
    device = torch.device(device)
    metric = TopKAccuracy(ct.TRAIN, topk=(1, 2, 3)).to(device=device)
    metric.attach(engine)
    metric(torch.eye(3, device=device), torch.arange(3, device=device))
    engine.fire_event(EngineEvents.TRAIN_EPOCH_STARTED)
    metric(
        prediction=torch.tensor([[0, 2, 1], [2, 1, 0]], device=device),
        target=torch.tensor([1, 2], device=device),
    )
    engine.fire_event(EngineEvents.TRAIN_EPOCH_COMPLETED)
    assert engine.get_history(f"{ct.TRAIN}/acc_top_1_accuracy").get_last_value() == 0.5
    assert engine.get_history(f"{ct.TRAIN}/acc_top_1_num_predictions").get_last_value() == 2
    assert engine.get_history(f"{ct.TRAIN}/acc_top_2_accuracy").get_last_value() == 0.5
    assert engine.get_history(f"{ct.TRAIN}/acc_top_2_num_predictions").get_last_value() == 2
    assert engine.get_history(f"{ct.TRAIN}/acc_top_3_accuracy").get_last_value() == 1.0
    assert engine.get_history(f"{ct.TRAIN}/acc_top_3_num_predictions").get_last_value() == 2


@mark.parametrize("device", get_available_devices())
def test_top_k_accuracy_events_eval(device: str, engine: BaseEngine) -> None:
    device = torch.device(device)
    metric = TopKAccuracy(ct.EVAL, topk=(1, 2, 3)).to(device=device)
    metric.attach(engine)
    metric(torch.eye(3, device=device), torch.arange(3, device=device))
    engine.fire_event(EngineEvents.EVAL_EPOCH_STARTED)
    metric(
        prediction=torch.tensor([[0, 2, 1], [2, 1, 0]], device=device),
        target=torch.tensor([1, 2], device=device),
    )
    engine.fire_event(EngineEvents.EVAL_EPOCH_COMPLETED)
    assert engine.get_history(f"{ct.EVAL}/acc_top_1_accuracy").get_last_value() == 0.5
    assert engine.get_history(f"{ct.EVAL}/acc_top_1_num_predictions").get_last_value() == 2
    assert engine.get_history(f"{ct.EVAL}/acc_top_2_accuracy").get_last_value() == 0.5
    assert engine.get_history(f"{ct.EVAL}/acc_top_2_num_predictions").get_last_value() == 2
    assert engine.get_history(f"{ct.EVAL}/acc_top_3_accuracy").get_last_value() == 1.0
    assert engine.get_history(f"{ct.EVAL}/acc_top_3_num_predictions").get_last_value() == 2


def test_top_k_accuracy_reset() -> None:
    metric = TopKAccuracy(ct.EVAL, topk=(1, 3))
    metric(prediction=torch.eye(4), target=torch.ones(4))
    metric.reset()
    assert metric._states[1].num_predictions == 0
    assert metric._states[3].num_predictions == 0
