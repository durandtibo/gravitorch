from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import torch
from objectory import OBJECT_TARGET
from pytest import mark, raises
from torch.backends import mps

from gravitorch import constants as ct
from gravitorch.engines import BaseEngine, EngineEvents
from gravitorch.events import VanillaEventHandler
from gravitorch.loops.evaluation import FabricEvaluationLoop
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
    lightning_available,
)
from gravitorch.utils import get_available_devices
from gravitorch.utils.exp_trackers import EpochStep
from gravitorch.utils.history import EmptyHistoryError, MinScalarHistory
from gravitorch.utils.imports import is_lightning_available
from gravitorch.utils.profilers import BaseProfiler, NoOpProfiler, PyTorchProfiler

if is_lightning_available():
    from lightning import Fabric
else:
    Fabric = None  # pragma: no cover

ACCELERATORS = ["auto", "cpu"]
if torch.cuda.is_available():
    ACCELERATORS.append("cuda")
if mps.is_available():
    ACCELERATORS.append("mps")


def increment_epoch_handler(engine: BaseEngine) -> None:
    engine.increment_epoch(2)


##########################################
#     Tests for FabricEvaluationLoop     #
##########################################


def test_fabric_evaluation_loop_str() -> None:
    assert str(FabricEvaluationLoop()).startswith("FabricEvaluationLoop(")


def test_fabric_evaluation_loop_missing_package() -> None:
    with patch("gravitorch.utils.imports.is_lightning_available", lambda *args: False):
        with raises(RuntimeError, match="`lightning` package is required but not installed."):
            FabricEvaluationLoop()


@mark.parametrize("tag", ("val", "test"))
def test_fabric_evaluation_loop_tag(tag: str) -> None:
    assert FabricEvaluationLoop(tag=tag)._tag == tag


def test_fabric_evaluation_loop_tag_default() -> None:
    assert FabricEvaluationLoop()._tag == "eval"


def test_fabric_evaluation_loop_condition() -> None:
    evaluation_loop = FabricEvaluationLoop(
        condition={OBJECT_TARGET: "gravitorch.loops.evaluation.conditions.LastEpochEvalCondition"}
    )
    assert isinstance(evaluation_loop._condition, LastEpochEvalCondition)


def test_fabric_evaluation_loop_condition_default() -> None:
    assert isinstance(FabricEvaluationLoop()._condition, EveryEpochEvalCondition)


def test_fabric_evaluation_loop_observer(tmp_path: Path) -> None:
    assert isinstance(
        FabricEvaluationLoop(observer=PyTorchBatchSaver(tmp_path))._observer,
        PyTorchBatchSaver,
    )


def test_fabric_evaluation_loop_observer_default() -> None:
    assert isinstance(FabricEvaluationLoop()._observer, NoOpLoopObserver)


def test_fabric_evaluation_loop_no_profiler() -> None:
    assert isinstance(FabricEvaluationLoop()._profiler, NoOpProfiler)


def test_fabric_evaluation_loop_profiler_tensorboard() -> None:
    assert isinstance(
        FabricEvaluationLoop(profiler=PyTorchProfiler(torch.profiler.profile()))._profiler,
        PyTorchProfiler,
    )


@lightning_available
@mark.parametrize("device", get_available_devices())
@mark.parametrize("accelerator", ACCELERATORS)
def test_fabric_evaluation_loop_eval(device: str, accelerator: str) -> None:
    device = torch.device(device)
    engine = create_dummy_engine(device=device)
    FabricEvaluationLoop(Fabric(accelerator=accelerator)).eval(engine)
    assert not engine.model.training
    assert engine.epoch == -1
    assert engine.iteration == -1
    loss_history = engine.get_history(f"eval/{ct.LOSS}")
    assert isinstance(loss_history, MinScalarHistory)
    assert isinstance(loss_history.get_last_value(), float)
    assert len(loss_history.get_recent_history()) == 1


@lightning_available
@mark.parametrize("device", get_available_devices())
@mark.parametrize("accelerator", ACCELERATORS)
def test_fabric_evaluation_loop_eval_loss_nan(device: str, accelerator: str) -> None:
    device = torch.device(device)
    engine = create_dummy_engine(model=DummyClassificationModel(loss_nan=True), device=device)
    FabricEvaluationLoop(Fabric(accelerator=accelerator)).eval(engine)
    assert engine.epoch == -1
    assert engine.iteration == -1
    with raises(EmptyHistoryError, match=f"'eval/{ct.LOSS}' history is empty."):
        engine.get_history(
            f"eval/{ct.LOSS}"
        ).get_last_value()  # The loss is not logged because it is NaN


@lightning_available
@mark.parametrize("device", get_available_devices())
@mark.parametrize("accelerator", ACCELERATORS)
def test_fabric_evaluation_loop_eval_with_loss_history(device: str, accelerator: str) -> None:
    device = torch.device(device)
    engine = create_dummy_engine(device=device)
    engine.add_history(MinScalarHistory(f"eval/{ct.LOSS}"))
    engine.log_metric(f"eval/{ct.LOSS}", 1, EpochStep(-1))
    FabricEvaluationLoop(Fabric(accelerator=accelerator)).eval(engine)
    loss_history = engine.get_history(f"eval/{ct.LOSS}")
    assert isinstance(loss_history, MinScalarHistory)
    assert isinstance(loss_history.get_last_value(), float)
    assert len(loss_history.get_recent_history()) == 2


def test_fabric_evaluation_loop_eval_no_dataset() -> None:
    engine = create_dummy_engine(data_source=Mock(has_data_loader=Mock(return_value=False)))
    FabricEvaluationLoop(Fabric(accelerator="cpu")).eval(engine)
    assert engine.epoch == -1
    assert engine.iteration == -1
    with raises(EmptyHistoryError, match=f"'eval/{ct.LOSS}' history is empty."):
        engine.get_history(
            f"eval/{ct.LOSS}"
        ).get_last_value()  # The loss is not logged because there is no batch


@lightning_available
@mark.parametrize("device", get_available_devices())
@mark.parametrize("accelerator", ACCELERATORS)
def test_fabric_evaluation_loop_eval_empty_map_dataset(device: str, accelerator: str) -> None:
    device = torch.device(device)
    engine = create_dummy_engine(
        data_source=DummyDataSource(eval_dataset=DummyDataset(num_examples=0)), device=device
    )
    FabricEvaluationLoop(Fabric(accelerator=accelerator)).eval(engine)
    assert engine.epoch == -1
    assert engine.iteration == -1
    with raises(EmptyHistoryError, match=f"'eval/{ct.LOSS}' history is empty."):
        # The loss is not logged because there is no batch
        engine.get_history(f"eval/{ct.LOSS}").get_last_value()


@lightning_available
@mark.parametrize("device", get_available_devices())
@mark.parametrize("accelerator", ACCELERATORS)
def test_fabric_evaluation_loop_eval_iterable_dataset(device: str, accelerator: str) -> None:
    device = torch.device(device)
    engine = create_dummy_engine(
        data_source=DummyDataSource(eval_dataset=DummyIterableDataset(), batch_size=2),
        device=device,
    )
    FabricEvaluationLoop(Fabric(accelerator=accelerator)).eval(engine)
    assert engine.epoch == -1
    assert engine.iteration == -1
    assert isinstance(engine.get_history(f"eval/{ct.LOSS}").get_last_value(), float)


@lightning_available
@mark.parametrize("device", get_available_devices())
@mark.parametrize("accelerator", ACCELERATORS)
def test_fabric_evaluation_loop_eval_empty_iterable_dataset(device: str, accelerator: str) -> None:
    device = torch.device(device)
    engine = create_dummy_engine(
        data_source=DummyDataSource(
            eval_dataset=DummyIterableDataset(num_examples=0), batch_size=None
        ),
        device=device,
    )
    FabricEvaluationLoop(Fabric(accelerator=accelerator)).eval(engine)
    assert engine.epoch == -1
    assert engine.iteration == -1
    with raises(EmptyHistoryError, match=f"'eval/{ct.LOSS}' history is empty."):
        engine.get_history(
            f"eval/{ct.LOSS}"
        ).get_last_value()  # The loss is not logged because there is no batch


@lightning_available
@mark.parametrize("device", get_available_devices())
def test_fabric_evaluation_loop_eval_skip_evaluation(device: str) -> None:
    device = torch.device(device)
    engine = create_dummy_engine(device=device)
    FabricEvaluationLoop(
        Fabric(accelerator="cpu"),
        condition=Mock(return_value=False),
    ).eval(engine)
    with raises(EmptyHistoryError, match=f"'eval/{ct.LOSS}' history is empty."):
        engine.get_history(
            f"eval/{ct.LOSS}"
        ).get_last_value()  # The loss is not logged because it is NaN


@mark.parametrize("device", get_available_devices())
@mark.parametrize("event", (EngineEvents.EVAL_EPOCH_STARTED, EngineEvents.EVAL_EPOCH_COMPLETED))
def test_fabric_evaluation_loop_fire_event_eval_epoch_events(device: str, event: str) -> None:
    device = torch.device(device)
    engine = create_dummy_engine(device=device)
    engine.add_event_handler(
        event, VanillaEventHandler(increment_epoch_handler, handler_kwargs={"engine": engine})
    )
    FabricEvaluationLoop(Fabric(accelerator="cpu")).eval(engine)
    assert engine.epoch == 1
    assert engine.iteration == -1


@mark.parametrize("device", get_available_devices())
@mark.parametrize(
    "event", (EngineEvents.EVAL_ITERATION_STARTED, EngineEvents.EVAL_ITERATION_COMPLETED)
)
def test_fabric_evaluation_loop_fire_event_eval_iteration_events(device: str, event: str) -> None:
    device = torch.device(device)
    engine = create_dummy_engine(device=device)
    engine.add_event_handler(
        event, VanillaEventHandler(increment_epoch_handler, handler_kwargs={"engine": engine})
    )
    FabricEvaluationLoop(Fabric(accelerator="cpu")).eval(engine)
    assert engine.epoch == 7
    assert engine.iteration == -1


@lightning_available
@mark.parametrize("device", get_available_devices())
def test_fabric_evaluation_loop_train_with_observer(device: str) -> None:
    device = torch.device(device)
    engine = create_dummy_engine(device=device)
    observer = Mock(spec=BaseLoopObserver)
    FabricEvaluationLoop(
        Fabric(accelerator="cpu"),
        observer=observer,
    ).eval(engine)
    observer.start.assert_called_once_with(engine)
    assert observer.update.call_count == 4
    observer.end.assert_called_once_with(engine)


@lightning_available
@mark.parametrize("device", get_available_devices())
def test_fabric_evaluation_loop_eval_with_profiler(device: str) -> None:
    device = torch.device(device)
    profiler = MagicMock(spec=BaseProfiler)
    FabricEvaluationLoop(Fabric(accelerator="cpu"), profiler=profiler).eval(
        engine=create_dummy_engine(device=device)
    )
    assert profiler.__enter__().step.call_count == 4


def test_fabric_evaluation_loop_load_state_dict() -> None:
    FabricEvaluationLoop().load_state_dict({})  # Verify it does not raise error


def test_fabric_evaluation_loop_state_dict() -> None:
    assert FabricEvaluationLoop().state_dict() == {}


def test_fabric_evaluation_loop_grad_enabled_false() -> None:
    engine = create_dummy_engine()
    out = FabricEvaluationLoop(Fabric(accelerator="cpu"), grad_enabled=False)._eval_one_batch(
        engine,
        engine.model,
        batch={
            ct.TARGET: torch.tensor([1, 2]),
            ct.INPUT: torch.ones(2, 4, requires_grad=True),
        },
    )
    assert not out[ct.LOSS].requires_grad


def test_fabric_evaluation_loop_grad_enabled_true() -> None:
    engine = create_dummy_engine()
    batch = {
        ct.TARGET: torch.tensor([1, 2]),
        ct.INPUT: torch.ones(2, 4, requires_grad=True),
    }
    out = FabricEvaluationLoop(Fabric(accelerator="cpu"), grad_enabled=True)._eval_one_batch(
        engine, engine.model, batch
    )
    assert out[ct.LOSS].requires_grad
    out[ct.LOSS].backward()
    assert torch.is_tensor(batch[ct.INPUT].grad)


def test_fabric_evaluation_loop_eval_one_batch_fired_events() -> None:
    engine = Mock(spec=BaseEngine)
    FabricEvaluationLoop(Fabric(accelerator="cpu"))._eval_one_batch(
        engine=engine,
        model=DummyClassificationModel(),
        batch={ct.INPUT: torch.ones(8, 4), ct.TARGET: torch.ones(8, dtype=torch.long)},
    )
    assert engine.fire_event.call_args_list == [
        ((EngineEvents.EVAL_ITERATION_STARTED,), {}),
        ((EngineEvents.EVAL_ITERATION_COMPLETED,), {}),
    ]
