from typing import Union
from unittest.mock import Mock

import torch
from coola import objects_are_allclose
from objectory import OBJECT_TARGET
from pytest import fixture, mark
from torch.nn import Flatten, Identity, Module

from gravitorch import constants as ct
from gravitorch.engines import BaseEngine, EngineEvents
from gravitorch.models.metrics import (
    AbsoluteError,
    BaseMetric,
    TransformedPredictionTarget,
)
from gravitorch.nn import Asinh, Log1p, Symlog
from gravitorch.testing import create_dummy_engine
from gravitorch.utils import get_available_devices
from gravitorch.utils.events import VanillaEventHandler
from gravitorch.utils.history import MinScalarHistory

MODES = (ct.TRAIN, ct.EVAL)
SIZES = (1, 2)


@fixture
def engine() -> BaseEngine:
    return create_dummy_engine()


#################################################
#     Tests for TransformedPredictionTarget     #
#################################################


@mark.parametrize(
    "metric",
    (
        AbsoluteError(ct.EVAL),
        {OBJECT_TARGET: "gravitorch.models.metrics.AbsoluteError", "mode": ct.EVAL},
    ),
)
def test_transformed_prediction_target_metric(metric: Union[BaseMetric, dict]) -> None:
    assert isinstance(TransformedPredictionTarget(metric).metric, AbsoluteError)


def test_transformed_prediction_target_prediction_transform_default() -> None:
    assert isinstance(
        TransformedPredictionTarget(AbsoluteError(ct.EVAL)).prediction_transform, Identity
    )


@mark.parametrize("prediction_transform", (Symlog(), {OBJECT_TARGET: "gravitorch.nn.Symlog"}))
def test_transformed_prediction_target_prediction_transform(
    prediction_transform: Union[Module, dict]
) -> None:
    assert isinstance(
        TransformedPredictionTarget(
            AbsoluteError(ct.EVAL), prediction_transform=prediction_transform
        ).prediction_transform,
        Symlog,
    )


def test_transformed_prediction_target_target_transform_default() -> None:
    assert isinstance(
        TransformedPredictionTarget(AbsoluteError(ct.EVAL)).target_transform, Identity
    )


@mark.parametrize("target_transform", (Symlog(), {OBJECT_TARGET: "gravitorch.nn.Symlog"}))
def test_transformed_prediction_target_target_transform(
    target_transform: Union[Module, dict]
) -> None:
    assert isinstance(
        TransformedPredictionTarget(
            AbsoluteError(ct.EVAL), target_transform=target_transform
        ).target_transform,
        Symlog,
    )


def test_transformed_prediction_target_attach_mock() -> None:
    engine = Mock(spec=BaseEngine)
    metric = Mock(spec=BaseMetric)
    assert TransformedPredictionTarget(metric).attach(engine) is None
    metric.attach.assert_called_once_with(engine)


def test_transformed_prediction_target_attach_abs_err(engine: BaseEngine) -> None:
    metric = TransformedPredictionTarget(AbsoluteError(ct.TRAIN))
    metric.attach(engine)
    assert isinstance(engine.get_history(f"{ct.TRAIN}/abs_err_mean"), MinScalarHistory)
    assert isinstance(engine.get_history(f"{ct.TRAIN}/abs_err_max"), MinScalarHistory)
    assert isinstance(engine.get_history(f"{ct.TRAIN}/abs_err_min"), MinScalarHistory)
    assert isinstance(engine.get_history(f"{ct.TRAIN}/abs_err_sum"), MinScalarHistory)
    assert engine.has_event_handler(
        VanillaEventHandler(metric.metric.reset), EngineEvents.TRAIN_EPOCH_STARTED
    )
    assert engine.has_event_handler(
        VanillaEventHandler(metric.metric.value, handler_kwargs={"engine": engine}),
        EngineEvents.TRAIN_EPOCH_COMPLETED,
    )


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("feature_size", SIZES)
def test_transformed_prediction_target_forward(
    device: str, mode: str, batch_size: int, feature_size: int
) -> None:
    device = torch.device(device)
    metric = TransformedPredictionTarget(AbsoluteError(mode)).to(device=device)
    metric(
        torch.ones(batch_size, feature_size, device=device),
        torch.ones(batch_size, feature_size, device=device),
    )
    assert metric.value() == {
        f"{mode}/abs_err_mean": 0.0,
        f"{mode}/abs_err_max": 0.0,
        f"{mode}/abs_err_min": 0.0,
        f"{mode}/abs_err_sum": 0.0,
        f"{mode}/abs_err_num_predictions": batch_size * feature_size,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_transformed_prediction_target_forward_symlog(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = TransformedPredictionTarget(
        metric=AbsoluteError(mode),
        prediction_transform=Symlog(),
        target_transform=Symlog(),
    ).to(device=device)
    metric(torch.ones(2, 2, device=device), torch.eye(2, device=device).mul(2).sub(1))
    assert objects_are_allclose(
        metric.value(),
        {
            f"{mode}/abs_err_mean": 0.6931471805599453,
            f"{mode}/abs_err_max": 1.3862943611198906,
            f"{mode}/abs_err_min": 0.0,
            f"{mode}/abs_err_sum": 2.772588722239781,
            f"{mode}/abs_err_num_predictions": 4,
        },
    )


@mark.parametrize("device", get_available_devices())
@mark.parametrize(
    "metric",
    (
        TransformedPredictionTarget.create_asinh(AbsoluteError(ct.EVAL)),
        TransformedPredictionTarget.create_flatten(AbsoluteError(ct.EVAL)),
        TransformedPredictionTarget.create_log1p(AbsoluteError(ct.EVAL)),
        TransformedPredictionTarget.create_symlog(AbsoluteError(ct.EVAL)),
    ),
)
def test_transformed_prediction_target_forward_transformations(
    device: str,
    metric: BaseMetric,
) -> None:
    device = torch.device(device)
    metric.reset()
    metric = metric.to(device=device)
    metric(torch.eye(2, device=device), torch.eye(2, device=device))
    assert metric.value() == {
        f"{ct.EVAL}/abs_err_mean": 0.0,
        f"{ct.EVAL}/abs_err_max": 0.0,
        f"{ct.EVAL}/abs_err_min": 0.0,
        f"{ct.EVAL}/abs_err_sum": 0.0,
        f"{ct.EVAL}/abs_err_num_predictions": 4,
    }


def test_transformed_prediction_target_reset_mock() -> None:
    metric = Mock(spec=BaseMetric)
    assert TransformedPredictionTarget(metric).reset() is None
    metric.reset.assert_called_once_with()


def test_transformed_prediction_target_reset_mse() -> None:
    metric = AbsoluteError(ct.EVAL)
    metric._state._num_predictions = 5
    TransformedPredictionTarget(metric).reset()
    assert metric._state.num_predictions == 0


def test_transformed_prediction_target_value_without_engine() -> None:
    metric = Mock(spec=BaseMetric, value=Mock(return_value={"metric": 42}))
    assert TransformedPredictionTarget(metric).value() == {"metric": 42}
    metric.value.assert_called_once_with(None)


def test_transformed_prediction_target_value_with_engine() -> None:
    engine = Mock(spec=BaseEngine)
    metric = Mock(spec=BaseMetric, value=Mock(return_value={"metric": 42}))
    assert TransformedPredictionTarget(metric).value(engine) == {"metric": 42}
    metric.value.assert_called_once_with(engine)


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_transformed_prediction_target_value_log_engine(
    device: str, mode: str, engine: BaseEngine
) -> None:
    device = torch.device(device)
    metric = TransformedPredictionTarget(AbsoluteError(mode)).to(device=device)
    metric(torch.eye(2, device=device), -torch.eye(2, device=device))
    metric.value(engine)
    assert engine.get_history(f"{mode}/abs_err_mean").get_last_value() == 1.0
    assert engine.get_history(f"{mode}/abs_err_max").get_last_value() == 2.0
    assert engine.get_history(f"{mode}/abs_err_min").get_last_value() == 0.0
    assert engine.get_history(f"{mode}/abs_err_sum").get_last_value() == 4.0
    assert engine.get_history(f"{mode}/abs_err_num_predictions").get_last_value() == 4


@mark.parametrize("device", get_available_devices())
def test_transformed_prediction_target_events_train(device: str, engine: BaseEngine) -> None:
    device = torch.device(device)
    metric = TransformedPredictionTarget(AbsoluteError(ct.TRAIN)).to(device=device)
    metric.attach(engine)
    metric(torch.ones(2, 2, device=device), torch.ones(2, 2, device=device))
    engine.fire_event(EngineEvents.TRAIN_EPOCH_STARTED)
    metric(torch.eye(2, device=device), -torch.eye(2, device=device))
    engine.fire_event(EngineEvents.TRAIN_EPOCH_COMPLETED)
    assert engine.get_history(f"{ct.TRAIN}/abs_err_mean").get_last_value() == 1.0
    assert engine.get_history(f"{ct.TRAIN}/abs_err_max").get_last_value() == 2.0
    assert engine.get_history(f"{ct.TRAIN}/abs_err_min").get_last_value() == 0.0
    assert engine.get_history(f"{ct.TRAIN}/abs_err_sum").get_last_value() == 4.0
    assert engine.get_history(f"{ct.TRAIN}/abs_err_num_predictions").get_last_value() == 4


@mark.parametrize("device", get_available_devices())
def test_transformed_prediction_target_events_eval(device: str, engine: BaseEngine) -> None:
    device = torch.device(device)
    metric = TransformedPredictionTarget(AbsoluteError(ct.EVAL)).to(device=device)
    metric.attach(engine)
    metric(torch.ones(2, 2, device=device), torch.ones(2, 2, device=device))
    engine.fire_event(EngineEvents.EVAL_EPOCH_STARTED)
    metric(torch.eye(2, device=device), -torch.eye(2, device=device))
    engine.fire_event(EngineEvents.EVAL_EPOCH_COMPLETED)
    assert engine.get_history(f"{ct.EVAL}/abs_err_mean").get_last_value() == 1.0
    assert engine.get_history(f"{ct.EVAL}/abs_err_max").get_last_value() == 2.0
    assert engine.get_history(f"{ct.EVAL}/abs_err_min").get_last_value() == 0.0
    assert engine.get_history(f"{ct.EVAL}/abs_err_sum").get_last_value() == 4.0
    assert engine.get_history(f"{ct.EVAL}/abs_err_num_predictions").get_last_value() == 4


def test_transformed_prediction_target_create_asinh() -> None:
    metric = TransformedPredictionTarget.create_asinh(AbsoluteError(ct.EVAL))
    assert isinstance(metric.prediction_transform, Asinh)
    assert isinstance(metric.target_transform, Asinh)


def test_transformed_prediction_target_create_flatten() -> None:
    metric = TransformedPredictionTarget.create_flatten(AbsoluteError(ct.EVAL))
    assert isinstance(metric.prediction_transform, Flatten)
    assert isinstance(metric.target_transform, Flatten)


def test_transformed_prediction_target_create_log1p() -> None:
    metric = TransformedPredictionTarget.create_log1p(AbsoluteError(ct.EVAL))
    assert isinstance(metric.prediction_transform, Log1p)
    assert isinstance(metric.target_transform, Log1p)


def test_transformed_prediction_target_create_symlog() -> None:
    metric = TransformedPredictionTarget.create_symlog(AbsoluteError(ct.EVAL))
    assert isinstance(metric.prediction_transform, Symlog)
    assert isinstance(metric.target_transform, Symlog)
