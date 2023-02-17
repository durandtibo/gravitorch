from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import torch
from objectory import OBJECT_TARGET
from pytest import fixture, mark, raises

from gravitorch import constants as ct
from gravitorch.engines import BaseEngine, EngineEvents
from gravitorch.loops.evaluation import AccelerateEvaluationLoop
from gravitorch.loops.evaluation.conditions import (
    EveryEpochEvalCondition,
    LastEpochEvalCondition,
)
from gravitorch.loops.observers import NoOpLoopObserver, PyTorchBatchSaver
from gravitorch.testing import (
    DummyClassificationModel,
    DummyDataSource,
    DummyIterableDataset,
    create_dummy_engine,
)
from gravitorch.utils.events import VanillaEventHandler
from gravitorch.utils.exp_trackers import EpochStep
from gravitorch.utils.history import EmptyHistoryError, MinScalarHistory
from gravitorch.utils.integrations import is_accelerate_available
from gravitorch.utils.profilers import NoOpProfiler, PyTorchProfiler
from tests.testing import accelerate_available

if is_accelerate_available():
    from accelerate import Accelerator
    from accelerate.state import AcceleratorState
else:
    Accelerator, AcceleratorState = None, None  # pragma: no cover


##############################################
#     Tests for AccelerateEvaluationLoop     #
##############################################


@fixture(autouse=True)
def reset_accelerate_state():
    if is_accelerate_available():
        AcceleratorState._shared_state = {}
        AcceleratorState.initialized = False


def increment_epoch_handler(engine: BaseEngine) -> None:
    engine.increment_epoch(2)


@accelerate_available
def test_accelerate_evaluation_loop_str():
    assert str(AccelerateEvaluationLoop()).startswith("AccelerateEvaluationLoop(")


def test_accelerate_evaluation_loop_missing_package():
    with patch("gravitorch.utils.integrations.is_accelerate_available", lambda *args: False):
        with raises(RuntimeError):
            AccelerateEvaluationLoop()


@accelerate_available
def test_accelerate_evaluation_loop_accelerator_none():
    assert isinstance(AccelerateEvaluationLoop()._accelerator, Accelerator)


@accelerate_available
def test_accelerate_evaluation_loop_accelerator_object():
    evaluation_loop = AccelerateEvaluationLoop(accelerator=Accelerator(cpu=True))
    assert isinstance(evaluation_loop._accelerator, Accelerator)
    assert evaluation_loop._accelerator.state.device.type == "cpu"


@accelerate_available
def test_accelerate_evaluation_loop_accelerator_from_dict():
    assert (
        AccelerateEvaluationLoop(accelerator={"cpu": True})._accelerator.state.device.type == "cpu"
    )


@accelerate_available
@mark.parametrize("tag", ("val", "test"))
def test_accelerate_evaluation_loop_tag(tag: str):
    assert AccelerateEvaluationLoop(tag=tag)._tag == tag


@accelerate_available
def test_accelerate_evaluation_loop_tag_default():
    assert AccelerateEvaluationLoop()._tag == "eval"


@accelerate_available
def test_accelerate_evaluation_loop_condition():
    evaluation_loop = AccelerateEvaluationLoop(
        condition={OBJECT_TARGET: "gravitorch.loops.evaluation.conditions.LastEpochEvalCondition"}
    )
    assert isinstance(evaluation_loop._condition, LastEpochEvalCondition)


@accelerate_available
def test_accelerate_evaluation_loop_condition_default():
    assert isinstance(AccelerateEvaluationLoop()._condition, EveryEpochEvalCondition)


@accelerate_available
def test_accelerate_evaluation_loop_observer(tmp_path: Path):
    assert isinstance(
        AccelerateEvaluationLoop(observer=PyTorchBatchSaver(tmp_path))._observer,
        PyTorchBatchSaver,
    )


@accelerate_available
def test_accelerate_evaluation_loop_observer_default():
    assert isinstance(AccelerateEvaluationLoop()._observer, NoOpLoopObserver)


@accelerate_available
def test_accelerate_evaluation_loop_no_profiler():
    assert isinstance(AccelerateEvaluationLoop()._profiler, NoOpProfiler)


@accelerate_available
def test_accelerate_evaluation_loop_profiler_tensorboard():
    assert isinstance(
        AccelerateEvaluationLoop(profiler=PyTorchProfiler(torch.profiler.profile()))._profiler,
        PyTorchProfiler,
    )


@accelerate_available
def test_accelerate_evaluation_loop_eval():
    engine = create_dummy_engine()
    evaluation_loop = AccelerateEvaluationLoop()
    evaluation_loop.eval(engine)
    assert not engine.model.training
    assert engine.epoch == -1
    assert engine.iteration == -1
    loss_history = engine.get_history(f"eval/{ct.LOSS}")
    assert isinstance(loss_history, MinScalarHistory)
    assert isinstance(loss_history.get_last_value(), float)
    assert len(loss_history.get_recent_history()) == 1


@accelerate_available
def test_accelerate_evaluation_loop_eval_loss_nan():
    engine = create_dummy_engine(model=DummyClassificationModel(loss_nan=True))
    evaluation_loop = AccelerateEvaluationLoop()
    evaluation_loop.eval(engine)
    assert engine.epoch == -1
    assert engine.iteration == -1
    with raises(EmptyHistoryError):
        engine.get_history(
            f"eval/{ct.LOSS}"
        ).get_last_value()  # The loss is not logged because it is NaN


@accelerate_available
def test_accelerate_evaluation_loop_eval_with_loss_history():
    engine = create_dummy_engine()
    engine.add_history(MinScalarHistory(f"eval/{ct.LOSS}"))
    engine.log_metric(f"eval/{ct.LOSS}", 1, EpochStep(-1))
    evaluation_loop = AccelerateEvaluationLoop()
    evaluation_loop.eval(engine)
    loss_history = engine.get_history(f"eval/{ct.LOSS}")
    assert isinstance(loss_history, MinScalarHistory)
    assert isinstance(loss_history.get_last_value(), float)
    assert len(loss_history.get_recent_history()) == 2


@accelerate_available
def test_accelerate_evaluation_loop_eval_no_dataset():
    engine = create_dummy_engine(data_source=Mock(has_data_loader=Mock(return_value=False)))
    evaluation_loop = AccelerateEvaluationLoop()
    evaluation_loop.eval(engine)
    assert engine.epoch == -1
    assert engine.iteration == -1
    with raises(EmptyHistoryError):
        engine.get_history(
            f"eval/{ct.LOSS}"
        ).get_last_value()  # The loss is not logged because there is no batch


# TODO: Comment this test because the current version of accelerate does not support
#  empty data loader
# @accelerate_available
# def test_accelerate_evaluation_loop_eval_empty_map_dataset():
#     engine = create_dummy_engine(data_source=FakeDataSource(eval_dataset=EmptyFakeMapDataset()))
#     evaluation_loop = AccelerateEvaluationLoop(accelerator={'dispatch_batches': False})
#     evaluation_loop.eval(engine)
#     assert engine.epoch == -1
#     assert engine.iteration == -1
#     with raises(EmptyHistoryError):
#         # The loss is not logged because there is no batch
#         engine.get_history(f"eval/{ct.LOSS}").get_last_value()


@accelerate_available
def test_accelerate_evaluation_loop_eval_iterable_dataset():
    engine = create_dummy_engine(
        data_source=DummyDataSource(eval_dataset=DummyIterableDataset(), batch_size=1)
    )
    evaluation_loop = AccelerateEvaluationLoop()
    evaluation_loop.eval(engine)
    assert engine.epoch == -1
    assert engine.iteration == -1
    assert isinstance(engine.get_history(f"eval/{ct.LOSS}").get_last_value(), float)


# TODO: Comment this test because the current version of accelerate does not support empty data loader
# @accelerate_available
# def test_accelerate_evaluation_loop_eval_empty_iterable_dataset():
#     engine = create_dummy_engine(
#         data_source=FakeDataSource(eval_dataset=EmptyFakeIterableDataset(), batch_size=None)
#     )
#     evaluation_loop = AccelerateEvaluationLoop(accelerator={"dispatch_batches": False})
#     evaluation_loop.eval(engine)
#     assert engine.epoch == -1
#     assert engine.iteration == -1
#     with raises(EmptyHistoryError):
#         # The loss is not logged because there is no batch
#         engine.get_history(f"eval/{ct.LOSS}").get_last_value()


@accelerate_available
def test_accelerate_evaluation_loop_eval_skip_evaluation():
    engine = create_dummy_engine()
    evaluation_loop = AccelerateEvaluationLoop(condition=Mock(return_value=False))
    evaluation_loop.eval(engine)
    with raises(EmptyHistoryError):
        engine.get_history(
            f"eval/{ct.LOSS}"
        ).get_last_value()  # The loss is not logged because it is NaN


@accelerate_available
@mark.parametrize("event", (EngineEvents.EVAL_EPOCH_STARTED, EngineEvents.EVAL_EPOCH_COMPLETED))
def test_accelerate_evaluation_loop_fire_event_eval_epoch_events(event):
    engine = create_dummy_engine()
    engine.add_event_handler(
        event, VanillaEventHandler(increment_epoch_handler, handler_kwargs={"engine": engine})
    )
    evaluation_loop = AccelerateEvaluationLoop()
    evaluation_loop.eval(engine)
    assert engine.epoch == 1
    assert engine.iteration == -1


@accelerate_available
@mark.parametrize(
    "event", (EngineEvents.EVAL_ITERATION_STARTED, EngineEvents.EVAL_ITERATION_COMPLETED)
)
def test_accelerate_evaluation_loop_fire_event_eval_iteration_events(event):
    engine = create_dummy_engine()
    engine.add_event_handler(
        event, VanillaEventHandler(increment_epoch_handler, handler_kwargs={"engine": engine})
    )
    evaluation_loop = AccelerateEvaluationLoop()
    evaluation_loop.eval(engine)
    assert engine.epoch == 7
    assert engine.iteration == -1


@accelerate_available
def test_accelerate_evaluation_loop_grad_enabled_false():
    engine = create_dummy_engine()
    loop = AccelerateEvaluationLoop(grad_enabled=False)
    batch = {
        ct.TARGET: torch.tensor([1, 2]),
        ct.INPUT: torch.ones(2, 4, requires_grad=True),
    }
    out = loop._eval_one_batch(engine, engine.model, batch)
    assert not out[ct.LOSS].requires_grad


@accelerate_available
def test_accelerate_evaluation_loop_grad_enabled_true():
    engine = create_dummy_engine()
    loop = AccelerateEvaluationLoop(grad_enabled=True)
    batch = {
        ct.TARGET: torch.tensor([1, 2]),
        ct.INPUT: torch.ones(2, 4, requires_grad=True),
    }
    out = loop._eval_one_batch(engine, engine.model, batch)
    assert out[ct.LOSS].requires_grad
    out[ct.LOSS].backward()
    assert torch.is_tensor(batch[ct.INPUT].grad)


@accelerate_available
def test_accelerate_evaluation_loop_train_with_observer():
    engine = create_dummy_engine()
    observer = MagicMock()
    AccelerateEvaluationLoop(observer=observer).eval(engine)
    observer.start.assert_called_once_with(engine)
    assert observer.update.call_count == 4
    observer.end.assert_called_once_with(engine)


@accelerate_available
def test_accelerate_evaluation_loop_eval_with_profiler():
    profiler = MagicMock()
    training_loop = AccelerateEvaluationLoop(profiler=profiler)
    training_loop.eval(engine=create_dummy_engine())
    assert profiler.__enter__().step.call_count == 4


@accelerate_available
def test_accelerate_evaluation_loop_load_state_dict():
    evaluation_loop = AccelerateEvaluationLoop()
    evaluation_loop.load_state_dict({})


@accelerate_available
def test_accelerate_evaluation_loop_state_dict():
    assert AccelerateEvaluationLoop().state_dict() == {}
