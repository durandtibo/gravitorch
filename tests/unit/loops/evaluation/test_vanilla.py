from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import torch
from objectory import OBJECT_TARGET
from pytest import mark, raises

from gravitorch import constants as ct
from gravitorch.engines import BaseEngine, EngineEvents
from gravitorch.loops.evaluation import EvaluationLoop
from gravitorch.loops.evaluation.conditions import (
    EveryEpochEvalCondition,
    LastEpochEvalCondition,
)
from gravitorch.loops.observers import (
    BaseLoopObserver,
    NoOpLoopObserver,
    PyTorchBatchSaver,
)
from gravitorch.testing import (
    DummyClassificationModel,
    DummyDataset,
    DummyDataSource,
    DummyIterableDataset,
    create_dummy_engine,
)
from gravitorch.utils import get_available_devices
from gravitorch.utils.device_placement import (
    AutoDevicePlacement,
    CpuDevicePlacement,
    ManualDevicePlacement,
)
from gravitorch.utils.events import GEventHandler
from gravitorch.utils.exp_trackers import EpochStep
from gravitorch.utils.history import EmptyHistoryError, MinScalarHistory
from gravitorch.utils.profilers import BaseProfiler, NoOpProfiler, PyTorchProfiler


def increment_epoch_handler(engine: BaseEngine) -> None:
    engine.increment_epoch(2)


####################################
#     Tests for EvaluationLoop     #
####################################


def test_evaluation_loop_str() -> None:
    assert str(EvaluationLoop()).startswith("EvaluationLoop(")


@mark.parametrize("tag", ("val", "test"))
def test_evaluation_loop_tag(tag: str) -> None:
    assert EvaluationLoop(tag=tag)._tag == tag


def test_evaluation_loop_tag_default() -> None:
    assert EvaluationLoop()._tag == "eval"


def test_evaluation_loop_batch_device_placement_cpu() -> None:
    assert isinstance(
        EvaluationLoop(batch_device_placement=CpuDevicePlacement())._batch_device_placement,
        CpuDevicePlacement,
    )


def test_evaluation_loop_batch_device_placement_default() -> None:
    assert isinstance(EvaluationLoop()._batch_device_placement, AutoDevicePlacement)


def test_evaluation_loop_condition() -> None:
    evaluation_loop = EvaluationLoop(
        condition={OBJECT_TARGET: "gravitorch.loops.evaluation.conditions.LastEpochEvalCondition"}
    )
    assert isinstance(evaluation_loop._condition, LastEpochEvalCondition)


def test_evaluation_loop_condition_default() -> None:
    assert isinstance(EvaluationLoop()._condition, EveryEpochEvalCondition)


def test_evaluation_loop_observer(tmp_path: Path) -> None:
    assert isinstance(
        EvaluationLoop(observer=PyTorchBatchSaver(tmp_path))._observer,
        PyTorchBatchSaver,
    )


def test_evaluation_loop_observer_default() -> None:
    assert isinstance(EvaluationLoop()._observer, NoOpLoopObserver)


def test_evaluation_loop_no_profiler() -> None:
    assert isinstance(EvaluationLoop()._profiler, NoOpProfiler)


def test_evaluation_loop_profiler_tensorboard() -> None:
    assert isinstance(
        EvaluationLoop(profiler=PyTorchProfiler(torch.profiler.profile()))._profiler,
        PyTorchProfiler,
    )


@mark.parametrize("device", get_available_devices())
def test_evaluation_loop_eval(device: str) -> None:
    device = torch.device(device)
    engine = create_dummy_engine(device=device)
    EvaluationLoop(batch_device_placement=ManualDevicePlacement(device)).eval(engine)
    assert not engine.model.training
    assert engine.epoch == -1
    assert engine.iteration == -1
    loss_history = engine.get_history(f"eval/{ct.LOSS}")
    assert isinstance(loss_history, MinScalarHistory)
    assert isinstance(loss_history.get_last_value(), float)
    assert len(loss_history.get_recent_history()) == 1


@mark.parametrize("device", get_available_devices())
def test_evaluation_loop_eval_loss_nan(device: str) -> None:
    device = torch.device(device)
    engine = create_dummy_engine(model=DummyClassificationModel(loss_nan=True), device=device)
    EvaluationLoop(batch_device_placement=ManualDevicePlacement(device)).eval(engine)
    assert engine.epoch == -1
    assert engine.iteration == -1
    with raises(EmptyHistoryError, match=f"'eval/{ct.LOSS}' history is empty."):
        engine.get_history(
            f"eval/{ct.LOSS}"
        ).get_last_value()  # The loss is not logged because it is NaN


@mark.parametrize("device", get_available_devices())
def test_evaluation_loop_eval_with_loss_history(device: str) -> None:
    device = torch.device(device)
    engine = create_dummy_engine(device=device)
    engine.add_history(MinScalarHistory(f"eval/{ct.LOSS}"))
    engine.log_metric(f"eval/{ct.LOSS}", 1, EpochStep(-1))
    EvaluationLoop(batch_device_placement=ManualDevicePlacement(device)).eval(engine)
    loss_history = engine.get_history(f"eval/{ct.LOSS}")
    assert isinstance(loss_history, MinScalarHistory)
    assert isinstance(loss_history.get_last_value(), float)
    assert len(loss_history.get_recent_history()) == 2


def test_evaluation_loop_eval_no_iterable() -> None:
    engine = create_dummy_engine(datasource=Mock(has_iterable=Mock(return_value=False)))
    EvaluationLoop().eval(engine)
    assert engine.epoch == -1
    assert engine.iteration == -1
    with raises(EmptyHistoryError, match=f"'eval/{ct.LOSS}' history is empty."):
        engine.get_history(
            f"eval/{ct.LOSS}"
        ).get_last_value()  # The loss is not logged because there is no batch


@mark.parametrize("device", get_available_devices())
def test_evaluation_loop_eval_empty_map_dataset(device: str) -> None:
    device = torch.device(device)
    engine = create_dummy_engine(
        datasource=DummyDataSource(eval_dataset=DummyDataset(num_examples=0)), device=device
    )
    EvaluationLoop(batch_device_placement=ManualDevicePlacement(device)).eval(engine)
    assert engine.epoch == -1
    assert engine.iteration == -1
    with raises(EmptyHistoryError, match=f"'eval/{ct.LOSS}' history is empty."):
        # The loss is not logged because there is no batch
        engine.get_history(f"eval/{ct.LOSS}").get_last_value()


@mark.parametrize("device", get_available_devices())
def test_evaluation_loop_eval_iterable_dataset(device: str) -> None:
    device = torch.device(device)
    engine = create_dummy_engine(
        datasource=DummyDataSource(eval_dataset=DummyIterableDataset(), batch_size=2),
        device=device,
    )
    EvaluationLoop(batch_device_placement=ManualDevicePlacement(device)).eval(engine)
    assert engine.epoch == -1
    assert engine.iteration == -1
    assert isinstance(engine.get_history(f"eval/{ct.LOSS}").get_last_value(), float)


@mark.parametrize("device", get_available_devices())
def test_evaluation_loop_eval_empty_iterable_dataset(device: str) -> None:
    device = torch.device(device)
    engine = create_dummy_engine(
        datasource=DummyDataSource(
            eval_dataset=DummyIterableDataset(num_examples=0), batch_size=None
        ),
        device=device,
    )
    EvaluationLoop(batch_device_placement=ManualDevicePlacement(device)).eval(engine)
    assert engine.epoch == -1
    assert engine.iteration == -1
    with raises(EmptyHistoryError, match=f"'eval/{ct.LOSS}' history is empty."):
        engine.get_history(
            f"eval/{ct.LOSS}"
        ).get_last_value()  # The loss is not logged because there is no batch


@mark.parametrize("device", get_available_devices())
def test_evaluation_loop_eval_auto_device_placement(device: str) -> None:
    device = torch.device(device)
    engine = create_dummy_engine(device=device)
    with patch("gravitorch.distributed.device", lambda *args, **kwargs: device):
        EvaluationLoop().eval(engine)
        assert not engine.model.training
        assert engine.epoch == -1
        assert engine.iteration == -1
        loss_history = engine.get_history(f"eval/{ct.LOSS}")
        assert isinstance(loss_history, MinScalarHistory)
        assert isinstance(loss_history.get_last_value(), float)
        assert len(loss_history.get_recent_history()) == 1


@mark.parametrize("device", get_available_devices())
def test_evaluation_loop_eval_skip_evaluation(device: str) -> None:
    device = torch.device(device)
    engine = create_dummy_engine(device=device)
    EvaluationLoop(
        condition=Mock(return_value=False), batch_device_placement=ManualDevicePlacement(device)
    ).eval(engine)
    with raises(EmptyHistoryError, match=f"'eval/{ct.LOSS}' history is empty."):
        engine.get_history(
            f"eval/{ct.LOSS}"
        ).get_last_value()  # The loss is not logged because it is NaN


@mark.parametrize("device", get_available_devices())
@mark.parametrize("event", (EngineEvents.EVAL_EPOCH_STARTED, EngineEvents.EVAL_EPOCH_COMPLETED))
def test_evaluation_loop_trigger_event_eval_epoch_events(device: str, event: str) -> None:
    device = torch.device(device)
    engine = create_dummy_engine(device=device)
    engine.add_event_handler(
        event, GEventHandler(increment_epoch_handler, handler_kwargs={"engine": engine})
    )
    EvaluationLoop(batch_device_placement=ManualDevicePlacement(device)).eval(engine)
    assert engine.epoch == 1
    assert engine.iteration == -1


@mark.parametrize("device", get_available_devices())
@mark.parametrize(
    "event", (EngineEvents.EVAL_ITERATION_STARTED, EngineEvents.EVAL_ITERATION_COMPLETED)
)
def test_evaluation_loop_trigger_event_eval_iteration_events(device: str, event: str) -> None:
    device = torch.device(device)
    engine = create_dummy_engine(device=device)
    engine.add_event_handler(
        event, GEventHandler(increment_epoch_handler, handler_kwargs={"engine": engine})
    )
    EvaluationLoop(batch_device_placement=ManualDevicePlacement(device)).eval(engine)
    assert engine.epoch == 7
    assert engine.iteration == -1


@mark.parametrize("device", get_available_devices())
def test_evaluation_loop_train_with_observer(device: str) -> None:
    device = torch.device(device)
    engine = create_dummy_engine(device=device)
    observer = MagicMock(spec=BaseLoopObserver)
    EvaluationLoop(observer=observer, batch_device_placement=ManualDevicePlacement(device)).eval(
        engine
    )
    observer.start.assert_called_once_with(engine)
    assert observer.update.call_count == 4
    observer.end.assert_called_once_with(engine)


@mark.parametrize("device", get_available_devices())
def test_evaluation_loop_eval_with_profiler(device: str) -> None:
    device = torch.device(device)
    profiler = MagicMock(spec=BaseProfiler)
    EvaluationLoop(profiler=profiler, batch_device_placement=ManualDevicePlacement(device)).eval(
        engine=create_dummy_engine(device=device)
    )
    assert profiler.__enter__().step.call_count == 4


def test_evaluation_loop_load_state_dict() -> None:
    EvaluationLoop().load_state_dict({})  # Verify it does not raise error


def test_evaluation_loop_state_dict() -> None:
    assert EvaluationLoop().state_dict() == {}


@mark.parametrize("device", get_available_devices())
def test_evaluation_loop_grad_enabled_false(device: str) -> None:
    device = torch.device(device)
    engine = create_dummy_engine(device=device)
    loop = EvaluationLoop(grad_enabled=False, batch_device_placement=ManualDevicePlacement(device))
    batch = {
        ct.TARGET: torch.tensor([1, 2]),
        ct.INPUT: torch.ones(2, 4, requires_grad=True),
    }
    out = loop._eval_one_batch(engine, engine.model, batch)
    assert not out[ct.LOSS].requires_grad


@mark.parametrize("device", get_available_devices())
def test_evaluation_loop_grad_enabled_true(device: str) -> None:
    device = torch.device(device)
    engine = create_dummy_engine(device=device)
    loop = EvaluationLoop(grad_enabled=True, batch_device_placement=ManualDevicePlacement(device))
    batch = {
        ct.TARGET: torch.tensor([1, 2]),
        ct.INPUT: torch.ones(2, 4, requires_grad=True),
    }
    out = loop._eval_one_batch(engine, engine.model, batch)
    assert out[ct.LOSS].requires_grad
    out[ct.LOSS].backward()
    assert torch.is_tensor(batch[ct.INPUT].grad)


@mark.parametrize("device", get_available_devices())
def test_evaluation_loop_eval_one_batch_fired_events(device: str) -> None:
    device = torch.device(device)
    engine = Mock(spec=BaseEngine)
    EvaluationLoop(batch_device_placement=ManualDevicePlacement(device))._eval_one_batch(
        engine=engine,
        model=DummyClassificationModel().to(device=device),
        batch={ct.INPUT: torch.ones(8, 4), ct.TARGET: torch.ones(8, dtype=torch.long)},
    )
    assert engine.trigger_event.call_args_list == [
        ((EngineEvents.EVAL_ITERATION_STARTED,), {}),
        ((EngineEvents.EVAL_ITERATION_COMPLETED,), {}),
    ]
