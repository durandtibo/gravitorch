from unittest.mock import Mock, patch

import torch
from pytest import mark

from gravitorch import constants as ct
from gravitorch.engines import BaseEngine, EngineEvents
from gravitorch.loops.evaluation import AMPEvaluationLoop
from gravitorch.testing import DummyClassificationModel
from gravitorch.utils import get_available_devices
from gravitorch.utils.device_placement import ManualDevicePlacement

#######################################
#     Tests for AMPEvaluationLoop     #
#######################################


def test_amp_evaluation_loop_str() -> None:
    assert str(AMPEvaluationLoop()).startswith("AMPEvaluationLoop(")


@mark.parametrize("amp_enabled", (True, False))
def test_amp_evaluation_loop_amp_enabled(amp_enabled: bool) -> None:
    assert AMPEvaluationLoop(amp_enabled=amp_enabled)._amp_enabled == amp_enabled


@mark.parametrize("device", get_available_devices())
def test_amp_evaluation_loop_eval_one_batch_fired_events(device: str) -> None:
    device = torch.device(device)
    engine = Mock(spec=BaseEngine)
    AMPEvaluationLoop(batch_device_placement=ManualDevicePlacement(device))._eval_one_batch(
        engine=engine,
        model=DummyClassificationModel().to(device=device),
        batch={ct.INPUT: torch.ones(8, 4), ct.TARGET: torch.ones(8, dtype=torch.long)},
    )
    assert engine.fire_event.call_args_list == [
        ((EngineEvents.EVAL_ITERATION_STARTED,), {}),
        ((EngineEvents.EVAL_ITERATION_COMPLETED,), {}),
    ]


@mark.parametrize("device", get_available_devices())
@mark.parametrize("amp_enabled", (True, False))
def test_amp_evaluation_loop_eval_one_batch_amp_enabled(device: str, amp_enabled: bool) -> None:
    device = torch.device(device)
    with patch("gravitorch.loops.evaluation.amp.autocast") as autocast_mock:
        AMPEvaluationLoop(
            amp_enabled=amp_enabled, batch_device_placement=ManualDevicePlacement(device)
        )._eval_one_batch(
            engine=Mock(spec=BaseEngine),
            model=DummyClassificationModel().to(device=device),
            batch={ct.INPUT: torch.ones(8, 4), ct.TARGET: torch.ones(8, dtype=torch.long)},
        )
        autocast_mock.assert_called_once_with(enabled=amp_enabled)
