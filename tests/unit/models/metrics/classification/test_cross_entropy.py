import math
from unittest.mock import Mock

import torch
from coola import objects_are_allclose
from pytest import fixture, mark, raises

from gravitorch import constants as ct
from gravitorch.engines import BaseEngine, EngineEvents
from gravitorch.models.metrics import CategoricalCrossEntropy, EmptyMetricError
from gravitorch.models.metrics.state import (
    BaseState,
    ExtendedErrorState,
    MeanErrorState,
)
from gravitorch.testing import create_dummy_engine
from gravitorch.utils import get_available_devices
from gravitorch.utils.events import VanillaEventHandler
from gravitorch.utils.history import MinScalarHistory

MODES = (ct.TRAIN, ct.EVAL)
NAMES = ("name1", "name2")
SIZES = (1, 2)
DTYPES = (torch.long, torch.float)

TOLERANCE = 1e-7


@fixture
def engine() -> BaseEngine:
    return create_dummy_engine()


#############################################
#     Tests for CategoricalCrossEntropy     #
#############################################


@mark.parametrize("mode", MODES)
def test_categorical_cross_entropy_str(mode: str) -> None:
    assert str(CategoricalCrossEntropy(mode)).startswith("CategoricalCrossEntropy(")


def test_categorical_cross_entropy_state_default() -> None:
    assert isinstance(CategoricalCrossEntropy(ct.EVAL)._state, MeanErrorState)


def test_categorical_cross_entropy_state_extended() -> None:
    assert isinstance(
        CategoricalCrossEntropy(ct.EVAL, state=ExtendedErrorState())._state,
        ExtendedErrorState,
    )


@mark.parametrize("name", NAMES)
def test_categorical_cross_entropy_attach_train(name: str, engine: BaseEngine):
    metric = CategoricalCrossEntropy(ct.TRAIN, name=name)
    metric.attach(engine)
    assert isinstance(engine.get_history(f"{ct.TRAIN}/{name}_mean"), MinScalarHistory)
    assert engine.has_event_handler(
        VanillaEventHandler(metric.reset), EngineEvents.TRAIN_EPOCH_STARTED
    )
    assert engine.has_event_handler(
        VanillaEventHandler(metric.value, handler_kwargs={"engine": engine}),
        EngineEvents.TRAIN_EPOCH_COMPLETED,
    )


@mark.parametrize("name", NAMES)
def test_categorical_cross_entropy_attach_eval(name: str, engine: BaseEngine):
    metric = CategoricalCrossEntropy(ct.EVAL, name=name)
    metric.attach(engine)
    assert isinstance(engine.get_history(f"{ct.EVAL}/{name}_mean"), MinScalarHistory)
    assert engine.has_event_handler(
        VanillaEventHandler(metric.reset), EngineEvents.EVAL_EPOCH_STARTED
    )
    assert engine.has_event_handler(
        VanillaEventHandler(metric.value, handler_kwargs={"engine": engine}),
        EngineEvents.EVAL_EPOCH_COMPLETED,
    )


def test_categorical_cross_entropy_attach_state_extended(engine: BaseEngine):
    metric = CategoricalCrossEntropy(ct.EVAL, state=ExtendedErrorState())
    metric.attach(engine)
    assert isinstance(engine.get_history(f"{ct.EVAL}/cat_ce_mean"), MinScalarHistory)
    assert isinstance(engine.get_history(f"{ct.EVAL}/cat_ce_median"), MinScalarHistory)
    assert isinstance(engine.get_history(f"{ct.EVAL}/cat_ce_min"), MinScalarHistory)
    assert isinstance(engine.get_history(f"{ct.EVAL}/cat_ce_max"), MinScalarHistory)
    assert isinstance(engine.get_history(f"{ct.EVAL}/cat_ce_sum"), MinScalarHistory)
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
def test_categorical_cross_entropy_forward_correct(device: str, mode: str, batch_size: int) -> None:
    device = torch.device(device)
    metric = CategoricalCrossEntropy(mode).to(device=device)
    prediction = torch.zeros(batch_size, 3, device=device)
    prediction[:, 0] = 1
    metric(prediction, torch.zeros(batch_size, device=device))
    assert objects_are_allclose(
        metric.value(),
        {
            f"{mode}/cat_ce_mean": 0.5514447139320511,
            f"{mode}/cat_ce_num_predictions": batch_size,
        },
        atol=TOLERANCE,
    )


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
@mark.parametrize("batch_size", SIZES)
def test_categorical_cross_entropy_forward_incorrect(
    device: str, mode: str, batch_size: int
) -> None:
    device = torch.device(device)
    metric = CategoricalCrossEntropy(mode).to(device=device)
    prediction = torch.zeros(batch_size, 3, device=device)
    prediction[:, 0] = 1
    metric(prediction, torch.ones(batch_size, device=device))
    assert objects_are_allclose(
        metric.value(),
        {
            f"{mode}/cat_ce_mean": 1.551444713932051,
            f"{mode}/cat_ce_num_predictions": batch_size,
        },
        atol=TOLERANCE,
    )


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_categorical_cross_entropy_forward_partially_correct(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = CategoricalCrossEntropy(mode).to(device=device)
    metric(torch.eye(2, device=device), torch.zeros(2, device=device))
    assert objects_are_allclose(
        metric.value(),
        {
            f"{mode}/cat_ce_mean": 0.8132616875182228,
            f"{mode}/cat_ce_num_predictions": 2,
        },
        atol=TOLERANCE,
    )


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_categorical_cross_entropy_forward_prediction_2d(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = CategoricalCrossEntropy(mode).to(device=device)
    metric(torch.ones(2, 3, device=device), torch.zeros(2, device=device))
    assert objects_are_allclose(
        metric.value(),
        {
            f"{mode}/cat_ce_mean": 1.0986122886681098,
            f"{mode}/cat_ce_num_predictions": 2,
        },
        atol=TOLERANCE,
    )


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_categorical_cross_entropy_forward_prediction_2d_target_2d(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = CategoricalCrossEntropy(mode).to(device=device)
    metric(torch.ones(2, 3, device=device), torch.zeros(2, 1, device=device))
    assert objects_are_allclose(
        metric.value(),
        {
            f"{mode}/cat_ce_mean": 1.0986122886681098,
            f"{mode}/cat_ce_num_predictions": 2,
        },
        atol=TOLERANCE,
    )


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_categorical_cross_entropy_forward_prediction_3d(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = CategoricalCrossEntropy(mode).to(device=device)
    metric(torch.ones(2, 3, 3, device=device), torch.zeros(2, 3, device=device))
    assert objects_are_allclose(
        metric.value(),
        {
            f"{mode}/cat_ce_mean": 1.0986122886681098,
            f"{mode}/cat_ce_num_predictions": 6,
        },
        atol=TOLERANCE,
    )


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_categorical_cross_entropy_forward_prediction_4d(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = CategoricalCrossEntropy(mode).to(device=device)
    metric(torch.ones(2, 3, 4, 3, device=device), torch.zeros(2, 3, 4, device=device))
    assert objects_are_allclose(
        metric.value(),
        {
            f"{mode}/cat_ce_mean": 1.0986122886681098,
            f"{mode}/cat_ce_num_predictions": 24,
        },
        atol=TOLERANCE,
    )


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
@mark.parametrize("dtype_target", DTYPES)
def test_categorical_cross_entropy_forward_dtypes(
    device: str,
    mode: str,
    dtype_target: torch.dtype,
):
    device = torch.device(device)
    metric = CategoricalCrossEntropy(mode).to(device=device)
    metric(torch.eye(4, device=device), torch.arange(4, device=device, dtype=dtype_target))
    assert objects_are_allclose(
        metric.value(),
        {
            f"{mode}/cat_ce_mean": 0.7436683806286791,
            f"{mode}/cat_ce_num_predictions": 4,
        },
        atol=TOLERANCE,
    )


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_categorical_cross_entropy_forward_state(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = CategoricalCrossEntropy(mode, state=ExtendedErrorState()).to(device=device)
    metric(torch.eye(4, device=device), torch.arange(4, device=device))
    assert objects_are_allclose(
        metric.value(),
        {
            f"{mode}/cat_ce_mean": 0.7436683806286791,
            f"{mode}/cat_ce_median": 0.7436683806286791,
            f"{mode}/cat_ce_min": 0.7436683806286791,
            f"{mode}/cat_ce_max": 0.7436683806286791,
            f"{mode}/cat_ce_sum": 2.9746735225147165,
            f"{mode}/cat_ce_std": 0.0,
            f"{mode}/cat_ce_num_predictions": 4,
        },
        atol=TOLERANCE,
    )


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_categorical_cross_entropy_forward_multiple_batches(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = CategoricalCrossEntropy(mode).to(device=device)
    metric(torch.eye(4, device=device), torch.arange(4, device=device))
    metric(torch.ones(2, 3, device=device), torch.ones(2, device=device))
    assert objects_are_allclose(
        metric.value(),
        {
            f"{mode}/cat_ce_mean": 0.8619830166418226,
            f"{mode}/cat_ce_num_predictions": 6,
        },
        atol=TOLERANCE,
    )


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_categorical_cross_entropy_forward_multiple_batches_with_reset(
    device: str, mode: str
) -> None:
    device = torch.device(device)
    metric = CategoricalCrossEntropy(mode).to(device=device)
    metric(torch.eye(4, device=device), torch.arange(4, device=device))
    metric.reset()
    metric(torch.ones(2, 3, device=device), torch.ones(2, device=device))
    assert objects_are_allclose(
        metric.value(),
        {
            f"{mode}/cat_ce_mean": 1.0986122886681098,
            f"{mode}/cat_ce_num_predictions": 2,
        },
        atol=TOLERANCE,
    )


@mark.parametrize("mode", MODES)
def test_categorical_cross_entropy_value_empty(mode):
    with raises(EmptyMetricError):
        CategoricalCrossEntropy(mode).value()


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_categorical_cross_entropy_value_log_engine(device: str, mode: str, engine: BaseEngine):
    device = torch.device(device)
    metric = CategoricalCrossEntropy(mode).to(device=device)
    metric(torch.eye(4, device=device), torch.arange(4, device=device))
    metric.value(engine)
    assert math.isclose(
        engine.get_history(f"{mode}/cat_ce_mean").get_last_value(),
        0.7436683806286791,
        abs_tol=TOLERANCE,
    )
    assert engine.get_history(f"{mode}/cat_ce_num_predictions").get_last_value() == 4


@mark.parametrize("device", get_available_devices())
def test_categorical_cross_entropy_events_train(device: str, engine: BaseEngine):
    device = torch.device(device)
    metric = CategoricalCrossEntropy(ct.TRAIN).to(device=device)
    metric.attach(engine)
    metric(torch.eye(4, device=device), torch.arange(4, device=device))
    engine.fire_event(EngineEvents.TRAIN_EPOCH_STARTED)
    metric(torch.ones(2, 3, device=device), torch.ones(2, device=device))
    engine.fire_event(EngineEvents.TRAIN_EPOCH_COMPLETED)
    assert math.isclose(
        engine.get_history(f"{ct.TRAIN}/cat_ce_mean").get_last_value(),
        1.0986122886681098,
        abs_tol=TOLERANCE,
    )
    assert engine.get_history(f"{ct.TRAIN}/cat_ce_num_predictions").get_last_value() == 2


@mark.parametrize("device", get_available_devices())
def test_categorical_cross_entropy_events_eval(device: str, engine: BaseEngine):
    device = torch.device(device)
    metric = CategoricalCrossEntropy(ct.EVAL).to(device=device)
    metric.attach(engine)
    metric(torch.eye(4, device=device), torch.arange(4, device=device))
    engine.fire_event(EngineEvents.EVAL_EPOCH_STARTED)
    metric(torch.ones(2, 3, device=device), torch.ones(2, device=device))
    engine.fire_event(EngineEvents.EVAL_EPOCH_COMPLETED)
    assert math.isclose(
        engine.get_history(f"{ct.EVAL}/cat_ce_mean").get_last_value(),
        1.0986122886681098,
        abs_tol=TOLERANCE,
    )
    assert engine.get_history(f"{ct.EVAL}/cat_ce_num_predictions").get_last_value() == 2


def test_categorical_cross_entropy_reset() -> None:
    state = Mock(spec=BaseState)
    metric = CategoricalCrossEntropy(ct.EVAL, state=state)
    metric.reset()
    state.reset.assert_called_once_with()
