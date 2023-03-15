import math
from unittest.mock import Mock, patch

import torch
from pytest import mark
from torch import nn
from torch.optim import SGD, Optimizer

from gravitorch import constants as ct
from gravitorch.engines import BaseEngine, EngineEvents
from gravitorch.loops.training import AMPTrainingLoop
from gravitorch.testing import DummyClassificationModel
from gravitorch.utils import get_available_devices
from gravitorch.utils.device_placement import ManualDevicePlacement

#####################################
#     Tests for AMPTrainingLoop     #
#####################################


def test_amp_training_loop_str() -> None:
    assert str(AMPTrainingLoop()).startswith("AMPTrainingLoop(")


@mark.parametrize("set_grad_to_none", (True, False))
def test_amp_training_loop_set_grad_to_none(set_grad_to_none: bool) -> None:
    assert AMPTrainingLoop(set_grad_to_none=set_grad_to_none)._set_grad_to_none == set_grad_to_none


def test_amp_training_loop_set_grad_to_none_default() -> None:
    assert AMPTrainingLoop()._set_grad_to_none


@mark.parametrize("amp_enabled", (True, False))
def test_amp_training_loop_amp_enabled(amp_enabled: bool) -> None:
    with patch("gravitorch.loops.training.amp.GradScaler") as scaler_mock:
        assert AMPTrainingLoop(amp_enabled=amp_enabled)._amp_enabled == amp_enabled
        scaler_mock.assert_called_once_with(enabled=amp_enabled)


def test_amp_training_loop_load_state_dict() -> None:
    AMPTrainingLoop(amp_enabled=False).load_state_dict({ct.SCALER: {}})


def test_amp_training_loop_state_dict() -> None:
    assert ct.SCALER in AMPTrainingLoop().state_dict()


@mark.parametrize("device", get_available_devices())
def test_amp_training_loop_train_one_batch_fired_events(device: str) -> None:
    device = torch.device(device)
    engine = Mock(spec=BaseEngine)
    model = DummyClassificationModel().to(device=device)
    AMPTrainingLoop(
        amp_enabled=False, batch_device_placement=ManualDevicePlacement(device)
    )._train_one_batch(
        engine=engine,
        model=model,
        optimizer=SGD(model.parameters(), lr=0.01),
        batch={ct.INPUT: torch.ones(8, 4), ct.TARGET: torch.ones(8, dtype=torch.long)},
    )
    assert engine.fire_event.call_args_list == [
        ((EngineEvents.TRAIN_ITERATION_STARTED,), {}),
        ((EngineEvents.TRAIN_FORWARD_COMPLETED,), {}),
        ((EngineEvents.TRAIN_BACKWARD_COMPLETED,), {}),
        ((EngineEvents.TRAIN_ITERATION_COMPLETED,), {}),
    ]


@mark.parametrize("device", get_available_devices())
@mark.parametrize("amp_enabled", (True, False))
def test_amp_training_loop_train_one_batch_amp_enabled(device: str, amp_enabled: bool) -> None:
    device = torch.device(device)
    model = DummyClassificationModel().to(device=device)
    with patch("torch.cuda.is_available", lambda *args, **kwargs: device.type == "cuda"):
        training_loop = AMPTrainingLoop(
            amp_enabled=amp_enabled,
            batch_device_placement=ManualDevicePlacement(device),
        )
        with patch("gravitorch.loops.training.amp.autocast") as autocast_mock:
            training_loop._train_one_batch(
                engine=Mock(spec=BaseEngine),
                model=model,
                optimizer=SGD(model.parameters(), lr=0.01),
                batch={ct.INPUT: torch.ones(8, 4), ct.TARGET: torch.ones(8, dtype=torch.long)},
            )
            autocast_mock.assert_called_once_with(enabled=amp_enabled)


@mark.parametrize("device", get_available_devices())
@mark.parametrize("set_grad_to_none", (True, False))
def test_amp_training_loop_train_one_batch_set_grad_to_none(
    device: str, set_grad_to_none: bool
) -> None:
    device = torch.device(device)
    engine = Mock(spec=BaseEngine)
    model = DummyClassificationModel().to(device=device)
    out = AMPTrainingLoop(
        set_grad_to_none=set_grad_to_none,
        amp_enabled=False,
        batch_device_placement=ManualDevicePlacement(device),
    )._train_one_batch(
        engine=engine,
        model=model,
        optimizer=SGD(model.parameters(), lr=0.01),
        batch={ct.INPUT: torch.ones(8, 4), ct.TARGET: torch.ones(8, dtype=torch.long)},
    )
    assert isinstance(out, dict)
    assert torch.is_tensor(out[ct.LOSS])
    assert out[ct.LOSS].device == device


@mark.parametrize("device", get_available_devices())
def test_amp_training_loop_train_one_batch_clip_grad_value(device: str) -> None:
    device = torch.device(device)
    engine = Mock(spec=BaseEngine)
    model = DummyClassificationModel().to(device=device)
    out = AMPTrainingLoop(
        clip_grad={"name": "clip_grad_value", "clip_value": 0.25},
        amp_enabled=False,
        batch_device_placement=ManualDevicePlacement(device),
    )._train_one_batch(
        engine=engine,
        model=model,
        optimizer=SGD(model.parameters(), lr=0.01),
        batch={ct.INPUT: torch.ones(8, 4), ct.TARGET: torch.ones(8, dtype=torch.long)},
    )
    assert isinstance(out, dict)
    assert torch.is_tensor(out[ct.LOSS])
    assert out[ct.LOSS].device == device


@mark.parametrize("device", get_available_devices())
def test_amp_training_loop_train_one_batch_clip_grad_norm(device: str) -> None:
    device = torch.device(device)
    engine = Mock(spec=BaseEngine)
    model = DummyClassificationModel().to(device=device)
    out = AMPTrainingLoop(
        clip_grad={"name": "clip_grad_norm", "max_norm": 1, "norm_type": 2},
        amp_enabled=False,
        batch_device_placement=ManualDevicePlacement(device),
    )._train_one_batch(
        engine=engine,
        model=model,
        optimizer=SGD(model.parameters(), lr=0.01),
        batch={ct.INPUT: torch.ones(8, 4), ct.TARGET: torch.ones(8, dtype=torch.long)},
    )
    assert isinstance(out, dict)
    assert torch.is_tensor(out[ct.LOSS])
    assert out[ct.LOSS].device == device


def test_amp_training_loop_train_one_batch_loss_nan() -> None:
    engine = Mock(spec=BaseEngine)
    model = Mock(spec=nn.Module, return_value={ct.LOSS: torch.tensor(math.nan)})
    optimizer = Mock(spec=Optimizer)
    out = AMPTrainingLoop(
        clip_grad={"name": "clip_grad_norm", "max_norm": 1, "norm_type": 2}, amp_enabled=False
    )._train_one_batch(
        engine=engine,
        model=model,
        optimizer=optimizer,
        batch={ct.INPUT: torch.ones(8, 4), ct.TARGET: torch.ones(8, dtype=torch.long)},
    )
    assert isinstance(out, dict)
    assert torch.isnan(out[ct.LOSS])
    assert engine.fire_event.call_args_list == [
        ((EngineEvents.TRAIN_ITERATION_STARTED,), {}),
        ((EngineEvents.TRAIN_FORWARD_COMPLETED,), {}),
        ((EngineEvents.TRAIN_ITERATION_COMPLETED,), {}),
    ]
