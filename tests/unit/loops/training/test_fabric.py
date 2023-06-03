import math
from pathlib import Path
from unittest.mock import MagicMock, Mock

import torch
from pytest import mark, raises
from torch import nn
from torch.backends import mps
from torch.optim import SGD, Optimizer

from gravitorch import constants as ct
from gravitorch.engines import BaseEngine, EngineEvents
from gravitorch.events import VanillaEventHandler
from gravitorch.loops.observers import (
    BaseLoopObserver,
    NoOpLoopObserver,
    PyTorchBatchSaver,
)
from gravitorch.loops.training import FabricTrainingLoop
from gravitorch.testing import (
    DummyClassificationModel,
    DummyDataset,
    DummyDataSource,
    DummyIterableDataset,
    create_dummy_engine,
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


#########################################
#     Tests for FabricTrainingLoop     #
#########################################


def test_fabric_training_loop_str() -> None:
    assert str(FabricTrainingLoop()).startswith("FabricTrainingLoop(")


@mark.parametrize("set_grad_to_none", (True, False))
def test_fabric_training_loop_set_grad_to_none(set_grad_to_none: bool) -> None:
    assert (
        FabricTrainingLoop(set_grad_to_none=set_grad_to_none)._set_grad_to_none == set_grad_to_none
    )


def test_fabric_training_loop_set_grad_to_none_default() -> None:
    assert FabricTrainingLoop()._set_grad_to_none


@mark.parametrize("tag", ("pre-training", "custom name"))
def test_fabric_training_loop_prefix(tag: str) -> None:
    assert FabricTrainingLoop(tag=tag)._tag == tag


def test_fabric_training_loop_prefix_default() -> None:
    assert FabricTrainingLoop()._tag == "train"


def test_fabric_training_loop_clip_grad_none() -> None:
    training_loop = FabricTrainingLoop()
    assert training_loop._clip_grad_fn is None
    assert training_loop._clip_grad_args == ()


def test_fabric_training_loop_clip_grad_clip_grad_value_without_clip_value() -> None:
    training_loop = FabricTrainingLoop(clip_grad={"name": "clip_grad_value"})
    assert callable(training_loop._clip_grad_fn)
    assert training_loop._clip_grad_args == (0.25,)


@mark.parametrize("clip_value", (0.1, 1))
def test_fabric_training_loop_clip_grad_clip_grad_value_with_clip_value(clip_value: float) -> None:
    training_loop = FabricTrainingLoop(
        clip_grad={"name": "clip_grad_value", "clip_value": clip_value}
    )
    assert callable(training_loop._clip_grad_fn)
    assert training_loop._clip_grad_args == (clip_value,)


def test_fabric_training_loop_clip_grad_clip_grad_norm_without_max_norm_and_norm_type() -> None:
    training_loop = FabricTrainingLoop(clip_grad={"name": "clip_grad_norm"})
    assert callable(training_loop._clip_grad_fn)
    assert training_loop._clip_grad_args == (1, 2)


@mark.parametrize("max_norm", (0.1, 1))
@mark.parametrize("norm_type", (1, 2))
def test_fabric_training_loop_clip_grad_clip_grad_norm_with_max_norm_and_norm_type(
    max_norm: float, norm_type: float
) -> None:
    training_loop = FabricTrainingLoop(
        clip_grad={"name": "clip_grad_norm", "max_norm": max_norm, "norm_type": norm_type}
    )
    assert callable(training_loop._clip_grad_fn)
    assert training_loop._clip_grad_args == (max_norm, norm_type)


def test_fabric_training_loop_clip_grad_incorrect_name() -> None:
    with raises(ValueError, match=r"Incorrect clip grad name \(incorrect name\)."):
        FabricTrainingLoop(clip_grad={"name": "incorrect name"})


def test_fabric_training_loop_observer_default() -> None:
    assert isinstance(FabricTrainingLoop()._observer, NoOpLoopObserver)


def test_fabric_training_loop_observer(tmp_path: Path) -> None:
    assert isinstance(
        FabricTrainingLoop(observer=PyTorchBatchSaver(tmp_path))._observer,
        PyTorchBatchSaver,
    )


def test_fabric_training_loop_no_profiler() -> None:
    assert isinstance(FabricTrainingLoop()._profiler, NoOpProfiler)


def test_fabric_training_loop_profiler_tensorboard() -> None:
    assert isinstance(
        FabricTrainingLoop(profiler=PyTorchProfiler(torch.profiler.profile()))._profiler,
        PyTorchProfiler,
    )


@mark.parametrize("device", get_available_devices())
@mark.parametrize("accelerator", ACCELERATORS)
def test_fabric_training_loop_train(device: str, accelerator: str) -> None:
    device = torch.device(device)
    engine = create_dummy_engine(device=device)
    FabricTrainingLoop(Fabric(accelerator=accelerator)).train(engine)
    assert engine.model.training
    assert engine.epoch == -1
    assert engine.iteration == 3
    loss_history = engine.get_history(f"train/{ct.LOSS}")
    assert isinstance(loss_history, MinScalarHistory)
    assert isinstance(loss_history.get_last_value(), float)
    assert len(loss_history.get_recent_history()) == 1


@mark.parametrize("device", get_available_devices())
@mark.parametrize("accelerator", ACCELERATORS)
def test_fabric_training_loop_train_loss_nan(device: str, accelerator: str) -> None:
    device = torch.device(device)
    engine = create_dummy_engine(model=DummyClassificationModel(loss_nan=True), device=device)
    FabricTrainingLoop(Fabric(accelerator=accelerator)).train(engine)
    assert engine.epoch == -1
    assert engine.iteration == 3
    with raises(EmptyHistoryError, match=f"'train/{ct.LOSS}' history is empty."):
        engine.get_history(
            f"train/{ct.LOSS}"
        ).get_last_value()  # The loss is not logged because it is NaN


@mark.parametrize("device", get_available_devices())
@mark.parametrize("accelerator", ACCELERATORS)
def test_fabric_training_loop_train_with_loss_history(device: str, accelerator: str) -> None:
    device = torch.device(device)
    engine = create_dummy_engine(device=device)
    engine.add_history(MinScalarHistory(f"train/{ct.LOSS}"))
    engine.log_metric(f"train/{ct.LOSS}", 1, EpochStep(-1))
    FabricTrainingLoop(Fabric(accelerator=accelerator)).train(engine)
    loss_history = engine.get_history(f"train/{ct.LOSS}")
    assert isinstance(loss_history, MinScalarHistory)
    assert isinstance(loss_history.get_last_value(), float)
    assert len(loss_history.get_recent_history()) == 2


@mark.parametrize("device", get_available_devices())
@mark.parametrize("accelerator", ACCELERATORS)
def test_fabric_training_loop_train_set_grad_to_none_true(device: str, accelerator: str) -> None:
    device = torch.device(device)
    engine = create_dummy_engine(device=device)
    FabricTrainingLoop(
        Fabric(accelerator=accelerator),
        set_grad_to_none=True,
    ).train(engine)
    assert engine.model.training
    assert engine.epoch == -1
    assert engine.iteration == 3
    loss_history = engine.get_history(f"train/{ct.LOSS}")
    assert isinstance(loss_history, MinScalarHistory)
    assert isinstance(loss_history.get_last_value(), float)


@mark.parametrize("device", get_available_devices())
@mark.parametrize("accelerator", ACCELERATORS)
def test_fabric_training_loop_train_with_clip_grad_value(device: str, accelerator: str) -> None:
    device = torch.device(device)
    engine = create_dummy_engine(device=device)
    FabricTrainingLoop(
        Fabric(accelerator=accelerator),
        clip_grad={"name": "clip_grad_value", "clip_value": 0.25},
    ).train(engine)
    assert engine.epoch == -1
    assert engine.iteration == 3
    assert isinstance(engine.get_history(f"train/{ct.LOSS}").get_last_value(), float)


@mark.parametrize("device", get_available_devices())
@mark.parametrize("accelerator", ACCELERATORS)
def test_fabric_training_loop_train_with_clip_grad_norm(device: str, accelerator: str) -> None:
    device = torch.device(device)
    engine = create_dummy_engine(device=device)
    FabricTrainingLoop(
        Fabric(accelerator=accelerator),
        clip_grad={"name": "clip_grad_norm", "max_norm": 1, "norm_type": 2},
    ).train(engine)
    assert engine.epoch == -1
    assert engine.iteration == 3
    assert isinstance(engine.get_history(f"train/{ct.LOSS}").get_last_value(), float)


@mark.parametrize("device", get_available_devices())
def test_fabric_training_loop_train_empty_map_dataset(device: str) -> None:
    device = torch.device(device)
    engine = create_dummy_engine(
        data_source=DummyDataSource(train_dataset=DummyDataset(num_examples=0)),
        device=device,
    )
    FabricTrainingLoop(Fabric(accelerator="cpu")).train(engine)
    assert engine.epoch == -1
    assert engine.iteration == -1
    with raises(EmptyHistoryError, match=f"'train/{ct.LOSS}' history is empty."):
        engine.get_history(
            f"train/{ct.LOSS}"
        ).get_last_value()  # The loss is not logged because there is no batch


@mark.parametrize("device", get_available_devices())
def test_fabric_training_loop_train_iterable_dataset(device: str) -> None:
    device = torch.device(device)
    engine = create_dummy_engine(
        data_source=DummyDataSource(train_dataset=DummyIterableDataset(), batch_size=2),
        device=device,
    )
    FabricTrainingLoop(Fabric(accelerator="cpu")).train(engine)
    assert engine.epoch == -1
    assert engine.iteration == 3
    assert isinstance(engine.get_history(f"train/{ct.LOSS}").get_last_value(), float)


@mark.parametrize("device", get_available_devices())
def test_fabric_training_loop_train_empty_iterable_dataset(device: str) -> None:
    device = torch.device(device)
    engine = create_dummy_engine(
        data_source=DummyDataSource(
            train_dataset=DummyIterableDataset(num_examples=0), batch_size=None
        ),
        device=device,
    )
    FabricTrainingLoop(Fabric(accelerator="cpu")).train(engine)
    assert engine.epoch == -1
    assert engine.iteration == -1
    with raises(EmptyHistoryError, match=f"'train/{ct.LOSS}' history is empty."):
        engine.get_history(
            f"train/{ct.LOSS}"
        ).get_last_value()  # The loss is not logged because there is no batch


@mark.parametrize("device", get_available_devices())
@mark.parametrize("event", (EngineEvents.TRAIN_EPOCH_STARTED, EngineEvents.TRAIN_EPOCH_COMPLETED))
def test_fabric_training_loop_fire_event_train_epoch_events(device: str, event: str) -> None:
    device = torch.device(device)
    engine = create_dummy_engine(device=device)
    engine.add_event_handler(
        event, VanillaEventHandler(increment_epoch_handler, handler_kwargs={"engine": engine})
    )
    FabricTrainingLoop(Fabric(accelerator="cpu")).train(engine)
    assert engine.epoch == 1
    assert engine.iteration == 3


@mark.parametrize("device", get_available_devices())
@mark.parametrize(
    "event",
    (
        EngineEvents.TRAIN_ITERATION_STARTED,
        EngineEvents.TRAIN_FORWARD_COMPLETED,
        EngineEvents.TRAIN_BACKWARD_COMPLETED,
        EngineEvents.TRAIN_ITERATION_COMPLETED,
    ),
)
def test_fabric_training_loop_fire_event_train_iteration_events(device: str, event: str) -> None:
    device = torch.device(device)
    engine = create_dummy_engine(device=device)
    engine.add_event_handler(
        event, VanillaEventHandler(increment_epoch_handler, handler_kwargs={"engine": engine})
    )
    FabricTrainingLoop(Fabric(accelerator="cpu")).train(engine)
    assert engine.epoch == 7
    assert engine.iteration == 3


@mark.parametrize("device", get_available_devices())
def test_fabric_training_loop_train_with_observer(device: str) -> None:
    device = torch.device(device)
    engine = create_dummy_engine(device=device)
    observer = Mock(spec=BaseLoopObserver)
    FabricTrainingLoop(
        Fabric(accelerator="cpu"),
        observer=observer,
    ).train(engine)
    observer.start.assert_called_once_with(engine)
    assert observer.update.call_count == 4
    observer.end.assert_called_once_with(engine)


@mark.parametrize("device", get_available_devices())
def test_fabric_training_loop_train_with_profiler(device: str) -> None:
    device = torch.device(device)
    profiler = MagicMock(spec=BaseProfiler)
    FabricTrainingLoop(
        Fabric(accelerator="cpu"),
        profiler=profiler,
    ).train(engine=create_dummy_engine(device=device))
    assert profiler.__enter__().step.call_count == 4


def test_fabric_training_loop_load_state_dict() -> None:
    FabricTrainingLoop().load_state_dict({})  # Verify it does not raise error


def test_fabric_training_loop_state_dict() -> None:
    assert FabricTrainingLoop().state_dict() == {}


@mark.parametrize("device", get_available_devices())
def test_fabric_training_loop_train_one_batch_fired_events(device: str) -> None:
    device = torch.device(device)
    engine = Mock(spec=BaseEngine)
    model = DummyClassificationModel().to(device=device)
    FabricTrainingLoop(Fabric(accelerator="cpu"))._train_one_batch(
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
@mark.parametrize("set_grad_to_none", (True, False))
def test_fabric_training_loop_train_one_batch_set_grad_to_none(
    device: str, set_grad_to_none: bool
) -> None:
    device = torch.device(device)
    engine = Mock(spec=BaseEngine)
    model = DummyClassificationModel().to(device=device)
    out = FabricTrainingLoop(
        Fabric(accelerator="cpu"),
        set_grad_to_none=set_grad_to_none,
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
def test_fabric_training_loop_train_one_batch_clip_grad_value(device: str) -> None:
    device = torch.device(device)
    engine = Mock(spec=BaseEngine)
    model = DummyClassificationModel().to(device=device)
    out = FabricTrainingLoop(
        Fabric(accelerator="cpu"),
        clip_grad={"name": "clip_grad_value", "clip_value": 0.25},
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
def test_fabric_training_loop_train_one_batch_clip_grad_norm(device: str) -> None:
    device = torch.device(device)
    engine = Mock(spec=BaseEngine)
    model = DummyClassificationModel().to(device=device)
    out = FabricTrainingLoop(
        Fabric(accelerator="cpu"),
        clip_grad={"name": "clip_grad_norm", "max_norm": 1, "norm_type": 2},
    )._train_one_batch(
        engine=engine,
        model=model,
        optimizer=SGD(model.parameters(), lr=0.01),
        batch={ct.INPUT: torch.ones(8, 4), ct.TARGET: torch.ones(8, dtype=torch.long)},
    )
    assert isinstance(out, dict)
    assert torch.is_tensor(out[ct.LOSS])
    assert out[ct.LOSS].device == device


def test_fabric_training_loop_train_one_batch_loss_nan() -> None:
    engine = Mock(spec=BaseEngine)
    model = Mock(spec=nn.Module, return_value={ct.LOSS: torch.tensor(math.nan)})
    optimizer = Mock(spec=Optimizer)
    out = FabricTrainingLoop(
        Fabric(accelerator="cpu"),
        clip_grad={"name": "clip_grad_norm", "max_norm": 1, "norm_type": 2},
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