from unittest.mock import Mock

import torch
from pytest import fixture, mark
from torch import nn

from gravitorch import constants as ct
from gravitorch.engines import BaseEngine, EngineEvents
from gravitorch.events import VanillaEventHandler
from gravitorch.models.metrics import AbsoluteError, SequentialMetric, SquaredError
from gravitorch.testing import create_dummy_engine
from gravitorch.utils import get_available_devices
from gravitorch.utils.history import MinScalarHistory


class FakeMetricWithOutput(nn.Module):
    def __init__(self, name: str) -> None:
        super().__init__()
        self._name = name

    def forward(self, *args, **kwargs) -> dict:
        return {self._name: 42}


@fixture
def metric() -> SequentialMetric:
    return SequentialMetric(metrics=[AbsoluteError(mode=ct.TRAIN), SquaredError(mode=ct.TRAIN)])


@fixture
def engine() -> BaseEngine:
    return create_dummy_engine()


######################################
#     Tests for SequentialMetric     #
######################################


def test_sequential_metric_attach() -> None:
    metrics = [Mock(spec=nn.Module, attach=Mock()), Mock(spec=nn.Module, attach=Mock())]
    metric = SequentialMetric(metrics)
    engine = Mock()
    metric.attach(engine)
    metrics[0].attach.assert_called_once_with(engine)
    metrics[1].attach.assert_called_once_with(engine)


def test_sequential_metric_attach_train(engine: BaseEngine) -> None:
    metric = SequentialMetric(metrics=[AbsoluteError(mode=ct.TRAIN), SquaredError(mode=ct.TRAIN)])
    metric.attach(engine)
    assert isinstance(engine.get_history(f"{ct.TRAIN}/abs_err_mean"), MinScalarHistory)
    assert isinstance(engine.get_history(f"{ct.TRAIN}/abs_err_max"), MinScalarHistory)
    assert isinstance(engine.get_history(f"{ct.TRAIN}/abs_err_min"), MinScalarHistory)
    assert isinstance(engine.get_history(f"{ct.TRAIN}/abs_err_sum"), MinScalarHistory)
    assert engine.has_event_handler(
        VanillaEventHandler(metric.metrics[0].reset), EngineEvents.TRAIN_EPOCH_STARTED
    )
    assert engine.has_event_handler(
        VanillaEventHandler(metric.metrics[0].value, handler_kwargs={"engine": engine}),
        EngineEvents.TRAIN_EPOCH_COMPLETED,
    )
    assert engine.has_event_handler(
        VanillaEventHandler(metric.metrics[1].reset), EngineEvents.TRAIN_EPOCH_STARTED
    )
    assert engine.has_event_handler(
        VanillaEventHandler(metric.metrics[1].value, handler_kwargs={"engine": engine}),
        EngineEvents.TRAIN_EPOCH_COMPLETED,
    )


def test_sequential_metric_attach_eval(engine: BaseEngine) -> None:
    metric = SequentialMetric(metrics=[AbsoluteError(mode=ct.EVAL), SquaredError(mode=ct.EVAL)])
    metric.attach(engine)
    assert isinstance(engine.get_history(f"{ct.EVAL}/abs_err_mean"), MinScalarHistory)
    assert isinstance(engine.get_history(f"{ct.EVAL}/abs_err_max"), MinScalarHistory)
    assert isinstance(engine.get_history(f"{ct.EVAL}/abs_err_min"), MinScalarHistory)
    assert isinstance(engine.get_history(f"{ct.EVAL}/abs_err_sum"), MinScalarHistory)
    assert engine.has_event_handler(
        VanillaEventHandler(metric.metrics[0].reset), EngineEvents.EVAL_EPOCH_STARTED
    )
    assert engine.has_event_handler(
        VanillaEventHandler(metric.metrics[0].value, handler_kwargs={"engine": engine}),
        EngineEvents.EVAL_EPOCH_COMPLETED,
    )
    assert engine.has_event_handler(
        VanillaEventHandler(metric.metrics[1].reset), EngineEvents.EVAL_EPOCH_STARTED
    )
    assert engine.has_event_handler(
        VanillaEventHandler(metric.metrics[1].value, handler_kwargs={"engine": engine}),
        EngineEvents.EVAL_EPOCH_COMPLETED,
    )


def test_sequential_metric_forward() -> None:
    metrics = [
        Mock(spec=nn.Module, return_value={"out1": 1}),
        Mock(spec=nn.Module, return_value={"out2": 2}),
    ]
    metric = SequentialMetric(metrics)
    features = Mock()
    assert metric(features) == {"out1": 1, "out2": 2}
    assert metrics[0].call_args.args == (features,)
    assert metrics[0].call_args.kwargs == {}
    assert metrics[1].call_args.args == (features,)
    assert metrics[1].call_args.kwargs == {}


def test_sequential_metric_forward_return_none() -> None:
    metrics = [
        Mock(spec=nn.Module, return_value=None),
        Mock(spec=nn.Module, return_value={"out2": 2}),
    ]
    metric = SequentialMetric(metrics)
    features = Mock()
    assert metric(features) == {"out2": 2}
    assert metrics[0].call_args.args == (features,)
    assert metrics[0].call_args.kwargs == {}
    assert metrics[1].call_args.args == (features,)
    assert metrics[1].call_args.kwargs == {}


@mark.parametrize("device", get_available_devices())
def test_sequential_metric_forward_no_output(device: str, metric: SequentialMetric) -> None:
    device = torch.device(device)
    metric.to(device=device)
    assert (
        metric(prediction=torch.ones(2, 3, device=device), target=torch.ones(2, 3, device=device))
        == {}
    )


def test_sequential_metric_forward_with_output() -> None:
    metric = SequentialMetric(
        metrics=[
            Mock(spec=nn.Module, return_value={"some": 42}),
            Mock(spec=nn.Module, return_value={"thing": 43}),
        ]
    )
    assert metric(prediction=torch.ones(2, 3), target=torch.ones(2, 3)) == {
        "some": 42,
        "thing": 43,
    }


@mark.parametrize("device", get_available_devices())
def test_sequential_metric_value(device: str, metric: SequentialMetric) -> None:
    device = torch.device(device)
    metric.to(device=device)
    metric(prediction=torch.ones(2, 3, device=device), target=torch.ones(2, 3, device=device))
    assert metric.value() == {
        f"{ct.TRAIN}/abs_err_mean": 0.0,
        f"{ct.TRAIN}/abs_err_max": 0.0,
        f"{ct.TRAIN}/abs_err_min": 0.0,
        f"{ct.TRAIN}/abs_err_sum": 0.0,
        f"{ct.TRAIN}/abs_err_num_predictions": 6,
        f"{ct.TRAIN}/sq_err_mean": 0.0,
        f"{ct.TRAIN}/sq_err_max": 0.0,
        f"{ct.TRAIN}/sq_err_min": 0.0,
        f"{ct.TRAIN}/sq_err_sum": 0.0,
        f"{ct.TRAIN}/sq_err_num_predictions": 6,
    }


@mark.parametrize("device", get_available_devices())
def test_sequential_metric_batches(device: str, metric: SequentialMetric) -> None:
    device = torch.device(device)
    metric.to(device=device)
    metric(prediction=torch.ones(2, 3, device=device), target=torch.ones(2, 3, device=device))
    metric(prediction=torch.zeros(2, 3, device=device), target=torch.ones(2, 3, device=device))
    assert metric.value() == {
        f"{ct.TRAIN}/abs_err_mean": 0.5,
        f"{ct.TRAIN}/abs_err_max": 1.0,
        f"{ct.TRAIN}/abs_err_min": 0.0,
        f"{ct.TRAIN}/abs_err_sum": 6.0,
        f"{ct.TRAIN}/abs_err_num_predictions": 12,
        f"{ct.TRAIN}/sq_err_mean": 0.5,
        f"{ct.TRAIN}/sq_err_max": 1.0,
        f"{ct.TRAIN}/sq_err_min": 0.0,
        f"{ct.TRAIN}/sq_err_sum": 6.0,
        f"{ct.TRAIN}/sq_err_num_predictions": 12,
    }


@mark.parametrize("device", get_available_devices())
def test_sequential_metric_batches_with_reset(device: str, metric: SequentialMetric) -> None:
    device = torch.device(device)
    metric.to(device=device)
    metric(prediction=torch.ones(2, 3, device=device), target=torch.ones(2, 3, device=device))
    metric.reset()
    metric(prediction=torch.zeros(2, 3, device=device), target=torch.ones(2, 3, device=device))
    assert metric.value() == {
        f"{ct.TRAIN}/abs_err_mean": 1.0,
        f"{ct.TRAIN}/abs_err_max": 1.0,
        f"{ct.TRAIN}/abs_err_min": 1.0,
        f"{ct.TRAIN}/abs_err_sum": 6.0,
        f"{ct.TRAIN}/abs_err_num_predictions": 6,
        f"{ct.TRAIN}/sq_err_mean": 1.0,
        f"{ct.TRAIN}/sq_err_max": 1.0,
        f"{ct.TRAIN}/sq_err_min": 1.0,
        f"{ct.TRAIN}/sq_err_sum": 6.0,
        f"{ct.TRAIN}/sq_err_num_predictions": 6,
    }


def test_sequential_metric_value_with_engine() -> None:
    metrics = [
        Mock(spec=nn.Module, value=Mock(return_value={"out1": 1})),
        Mock(spec=nn.Module, value=Mock(return_value={"out2": 2})),
    ]
    engine = Mock()
    assert SequentialMetric(metrics).value(engine) == {"out1": 1, "out2": 2}
    metrics[0].value.assert_called_once_with(engine)
    metrics[1].value.assert_called_once_with(engine)


def test_sequential_metric_value_without_engine() -> None:
    metrics = [
        Mock(spec=nn.Module, value=Mock(return_value={"out1": 1})),
        Mock(spec=nn.Module, value=Mock(return_value={"out2": 2})),
    ]
    assert SequentialMetric(metrics).value() == {"out1": 1, "out2": 2}
    metrics[0].value.assert_called_once_with(None)
    metrics[1].value.assert_called_once_with(None)
