from unittest.mock import Mock, patch

from pytest import mark, raises

from gravitorch.engines import BaseEngine, EngineEvents
from gravitorch.handlers import EpochCudaMemoryMonitor, IterationCudaMemoryMonitor
from gravitorch.utils.events import (
    ConditionalEventHandler,
    EpochPeriodicCondition,
    IterationPeriodicCondition,
)
from gravitorch.utils.exp_trackers import EpochStep, IterationStep

EVENTS = ("my_event", "my_other_event")


############################################
#     Tests for EpochCudaMemoryMonitor     #
############################################


def test_epoch_cuda_memory_monitor_str():
    assert str(EpochCudaMemoryMonitor()).startswith("EpochCudaMemoryMonitor(")


@mark.parametrize("event", EVENTS)
def test_epoch_cuda_memory_monitor_event(event: str):
    assert EpochCudaMemoryMonitor(event)._event == event


def test_epoch_cuda_memory_monitor_event_default():
    assert EpochCudaMemoryMonitor()._event == EngineEvents.EPOCH_COMPLETED


@mark.parametrize("freq", (1, 2))
def test_epoch_cuda_memory_monitor_freq(freq: int):
    assert EpochCudaMemoryMonitor(freq=freq)._freq == freq


@mark.parametrize("freq", (0, -1))
def test_epoch_cuda_memory_monitor_incorrect_freq(freq: int):
    with raises(ValueError):
        EpochCudaMemoryMonitor(freq=freq)


def test_epoch_cuda_memory_monitor_freq_default():
    assert EpochCudaMemoryMonitor()._freq == 1


@mark.parametrize("event", EVENTS)
@mark.parametrize("freq", (1, 2))
def test_epoch_cuda_memory_monitor_attach(event: str, freq: int):
    handler = EpochCudaMemoryMonitor(event=event, freq=freq)
    engine = Mock(spec=BaseEngine, epoch=-1, has_event_handler=Mock(return_value=False))
    handler.attach(engine)
    engine.add_event_handler.assert_called_once_with(
        event,
        ConditionalEventHandler(
            handler.monitor,
            condition=EpochPeriodicCondition(engine=engine, freq=freq),
            handler_kwargs={"engine": engine},
        ),
    )


def test_epoch_cuda_memory_monitor_attach_duplicate():
    engine = Mock(spec=BaseEngine, epoch=-1, has_event_handler=Mock(return_value=True))
    EpochCudaMemoryMonitor().attach(engine)
    engine.add_event_handler.assert_not_called()


@patch("gravitorch.utils.cudamem.torch.cuda.is_available", lambda *args: True)
@patch("gravitorch.utils.cudamem.torch.cuda.synchronize", lambda *args: None)
@patch("gravitorch.utils.cudamem.torch.cuda.mem_get_info", lambda *args: (None, 1))
def test_epoch_cuda_memory_monitor_monitor():
    engine = Mock(spec=BaseEngine, epoch=4)
    EpochCudaMemoryMonitor().monitor(engine)
    assert isinstance(engine.log_metrics.call_args.args[0]["epoch/max_cuda_memory_allocated"], int)
    assert isinstance(
        engine.log_metrics.call_args.args[0]["epoch/max_cuda_memory_allocated_pct"], float
    )
    assert engine.log_metrics.call_args.kwargs["step"] == EpochStep(4)


@patch("gravitorch.utils.cudamem.torch.cuda.is_available", lambda *args: False)
def test_epoch_cuda_memory_monitor_monitor_no_cuda():
    engine = Mock(spec=BaseEngine)
    EpochCudaMemoryMonitor().monitor(engine)
    engine.log_metric.assert_not_called()


################################################
#     Tests for IterationCudaMemoryMonitor     #
################################################


def test_iteration_cuda_memory_monitor_str():
    assert str(IterationCudaMemoryMonitor()).startswith("IterationCudaMemoryMonitor(")


@mark.parametrize("event", EVENTS)
def test_iteration_cuda_memory_monitor_event(event: str):
    assert IterationCudaMemoryMonitor(event)._event == event


def test_iteration_cuda_memory_monitor_event_default():
    assert IterationCudaMemoryMonitor()._event == EngineEvents.TRAIN_ITERATION_COMPLETED


@mark.parametrize("freq", (1, 2))
def test_iteration_cuda_memory_monitor_freq(freq: int):
    assert IterationCudaMemoryMonitor(freq=freq)._freq == freq


@mark.parametrize("freq", (0, -1))
def test_iteration_cuda_memory_monitor_incorrect_freq(freq: int):
    with raises(ValueError):
        IterationCudaMemoryMonitor(freq=freq)


def test_iteration_cuda_memory_monitor_freq_default():
    assert IterationCudaMemoryMonitor()._freq == 1


@mark.parametrize("event", EVENTS)
@mark.parametrize("freq", (1, 2))
def test_iteration_cuda_memory_monitor_attach(event: str, freq: int):
    handler = IterationCudaMemoryMonitor(event=event, freq=freq)
    engine = Mock(spec=BaseEngine, iteration=-1, has_event_handler=Mock(return_value=False))
    handler.attach(engine)
    engine.add_event_handler.assert_called_once_with(
        event,
        ConditionalEventHandler(
            handler.monitor,
            condition=IterationPeriodicCondition(engine=engine, freq=freq),
            handler_kwargs={"engine": engine},
        ),
    )


def test_iteration_cuda_memory_monitor_attach_duplicate():
    engine = Mock(spec=BaseEngine, iteration=-1, has_event_handler=Mock(return_value=True))
    IterationCudaMemoryMonitor().attach(engine)
    engine.add_event_handler.assert_not_called()


@patch("gravitorch.utils.cudamem.torch.cuda.is_available", lambda *args: True)
@patch("gravitorch.utils.cudamem.torch.cuda.synchronize", lambda *args: None)
@patch("gravitorch.utils.cudamem.torch.cuda.mem_get_info", lambda *args: (None, 1))
def test_iteration_cuda_memory_monitor_monitor():
    engine = Mock(spec=BaseEngine, iteration=4)
    IterationCudaMemoryMonitor().monitor(engine)
    assert isinstance(
        engine.log_metrics.call_args.args[0]["iteration/max_cuda_memory_allocated"], int
    )
    assert isinstance(
        engine.log_metrics.call_args.args[0]["iteration/max_cuda_memory_allocated_pct"], float
    )
    assert engine.log_metrics.call_args.kwargs["step"] == IterationStep(4)


@patch("gravitorch.utils.cudamem.torch.cuda.is_available", lambda *args: False)
def test_iteration_cuda_memory_monitor_monitor_no_cuda():
    engine = Mock(spec=BaseEngine)
    IterationCudaMemoryMonitor().monitor(engine)
    engine.log_metric.assert_not_called()
