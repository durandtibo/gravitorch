from unittest.mock import Mock, patch

from pytest import mark, raises

from gravitorch.engines import BaseEngine, EngineEvents
from gravitorch.events import (
    ConditionalEventHandler,
    EpochPeriodicCondition,
    IterationPeriodicCondition,
)
from gravitorch.handlers import (
    EpochCudaEmptyCache,
    EpochCudaMemoryMonitor,
    IterationCudaEmptyCache,
    IterationCudaMemoryMonitor,
)
from gravitorch.utils.exp_trackers import EpochStep, IterationStep

EVENTS = ("my_event", "my_other_event")


############################################
#     Tests for EpochCudaMemoryMonitor     #
############################################


def test_epoch_cuda_memory_monitor_str() -> None:
    assert str(EpochCudaMemoryMonitor()).startswith("EpochCudaMemoryMonitor(")


@mark.parametrize("event", EVENTS)
def test_epoch_cuda_memory_monitor_event(event: str) -> None:
    assert EpochCudaMemoryMonitor(event)._event == event


def test_epoch_cuda_memory_monitor_event_default() -> None:
    assert EpochCudaMemoryMonitor()._event == EngineEvents.EPOCH_COMPLETED


@mark.parametrize("freq", (1, 2))
def test_epoch_cuda_memory_monitor_freq(freq: int) -> None:
    assert EpochCudaMemoryMonitor(freq=freq)._freq == freq


@mark.parametrize("freq", (0, -1))
def test_epoch_cuda_memory_monitor_incorrect_freq(freq: int) -> None:
    with raises(ValueError, match="freq has to be greater than 0"):
        EpochCudaMemoryMonitor(freq=freq)


def test_epoch_cuda_memory_monitor_freq_default() -> None:
    assert EpochCudaMemoryMonitor()._freq == 1


@mark.parametrize("event", EVENTS)
@mark.parametrize("freq", (1, 2))
def test_epoch_cuda_memory_monitor_attach(event: str, freq: int) -> None:
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


def test_epoch_cuda_memory_monitor_attach_duplicate() -> None:
    engine = Mock(spec=BaseEngine, epoch=-1, has_event_handler=Mock(return_value=True))
    EpochCudaMemoryMonitor().attach(engine)
    engine.add_event_handler.assert_not_called()


@patch("torch.cuda.is_available", lambda *args: True)
@patch("torch.cuda.synchronize", lambda *args: None)
@patch("torch.cuda.mem_get_info", lambda *args: (None, 1))
def test_epoch_cuda_memory_monitor_monitor() -> None:
    engine = Mock(spec=BaseEngine, epoch=4)
    EpochCudaMemoryMonitor().monitor(engine)
    assert isinstance(engine.log_metrics.call_args.args[0]["epoch/max_cuda_memory_allocated"], int)
    assert isinstance(
        engine.log_metrics.call_args.args[0]["epoch/max_cuda_memory_allocated_pct"], float
    )
    assert engine.log_metrics.call_args.kwargs["step"] == EpochStep(4)


@patch("torch.cuda.is_available", lambda *args: False)
def test_epoch_cuda_memory_monitor_monitor_no_cuda() -> None:
    engine = Mock(spec=BaseEngine)
    EpochCudaMemoryMonitor().monitor(engine)
    engine.log_metric.assert_not_called()


################################################
#     Tests for IterationCudaMemoryMonitor     #
################################################


def test_iteration_cuda_memory_monitor_str() -> None:
    assert str(IterationCudaMemoryMonitor()).startswith("IterationCudaMemoryMonitor(")


@mark.parametrize("event", EVENTS)
def test_iteration_cuda_memory_monitor_event(event: str) -> None:
    assert IterationCudaMemoryMonitor(event)._event == event


def test_iteration_cuda_memory_monitor_event_default() -> None:
    assert IterationCudaMemoryMonitor()._event == EngineEvents.TRAIN_ITERATION_COMPLETED


@mark.parametrize("freq", (1, 2))
def test_iteration_cuda_memory_monitor_freq(freq: int) -> None:
    assert IterationCudaMemoryMonitor(freq=freq)._freq == freq


@mark.parametrize("freq", (0, -1))
def test_iteration_cuda_memory_monitor_incorrect_freq(freq: int) -> None:
    with raises(ValueError, match="freq has to be greater than 0"):
        IterationCudaMemoryMonitor(freq=freq)


def test_iteration_cuda_memory_monitor_freq_default() -> None:
    assert IterationCudaMemoryMonitor()._freq == 1


@mark.parametrize("event", EVENTS)
@mark.parametrize("freq", (1, 2))
def test_iteration_cuda_memory_monitor_attach(event: str, freq: int) -> None:
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


def test_iteration_cuda_memory_monitor_attach_duplicate() -> None:
    engine = Mock(spec=BaseEngine, iteration=-1, has_event_handler=Mock(return_value=True))
    IterationCudaMemoryMonitor().attach(engine)
    engine.add_event_handler.assert_not_called()


@patch("torch.cuda.is_available", lambda *args: True)
@patch("torch.cuda.synchronize", lambda *args: None)
@patch("torch.cuda.mem_get_info", lambda *args: (None, 1))
def test_iteration_cuda_memory_monitor_monitor() -> None:
    engine = Mock(spec=BaseEngine, iteration=4)
    IterationCudaMemoryMonitor().monitor(engine)
    assert isinstance(
        engine.log_metrics.call_args.args[0]["iteration/max_cuda_memory_allocated"], int
    )
    assert isinstance(
        engine.log_metrics.call_args.args[0]["iteration/max_cuda_memory_allocated_pct"], float
    )
    assert engine.log_metrics.call_args.kwargs["step"] == IterationStep(4)


@patch("torch.cuda.is_available", lambda *args: False)
def test_iteration_cuda_memory_monitor_monitor_no_cuda() -> None:
    engine = Mock(spec=BaseEngine)
    IterationCudaMemoryMonitor().monitor(engine)
    engine.log_metric.assert_not_called()


#########################################
#     Tests for EpochCudaEmptyCache     #
#########################################


def test_epoch_cuda_empty_cache_str() -> None:
    assert str(EpochCudaEmptyCache()).startswith("EpochCudaEmptyCache(")


@mark.parametrize("event", EVENTS)
def test_epoch_cuda_empty_cache_event(event: str) -> None:
    assert EpochCudaEmptyCache(event)._event == event


def test_epoch_cuda_empty_cache_event_default() -> None:
    assert EpochCudaEmptyCache()._event == EngineEvents.EPOCH_COMPLETED


@mark.parametrize("freq", (1, 2))
def test_epoch_cuda_empty_cache_freq(freq: int) -> None:
    assert EpochCudaEmptyCache(freq=freq)._freq == freq


@mark.parametrize("freq", (0, -1))
def test_epoch_cuda_empty_cache_incorrect_freq(freq: int) -> None:
    with raises(ValueError, match="freq has to be greater than 0"):
        EpochCudaEmptyCache(freq=freq)


def test_epoch_cuda_empty_cache_freq_default() -> None:
    assert EpochCudaEmptyCache()._freq == 1


@mark.parametrize("event", EVENTS)
@mark.parametrize("freq", (1, 2))
def test_epoch_cuda_empty_cache_attach(event: str, freq: int) -> None:
    handler = EpochCudaEmptyCache(event=event, freq=freq)
    engine = Mock(spec=BaseEngine, epoch=-1, has_event_handler=Mock(return_value=False))
    handler.attach(engine)
    engine.add_event_handler.assert_called_once_with(
        event,
        ConditionalEventHandler(
            handler.empty_cache,
            condition=EpochPeriodicCondition(engine=engine, freq=freq),
            handler_kwargs={"engine": engine},
        ),
    )


def test_epoch_cuda_empty_cache_attach_duplicate() -> None:
    engine = Mock(spec=BaseEngine, epoch=-1, has_event_handler=Mock(return_value=True))
    EpochCudaEmptyCache().attach(engine)
    engine.add_event_handler.assert_not_called()


@patch("torch.cuda.is_available", lambda *args: True)
def test_epoch_cuda_empty_cache_monitor_cuda() -> None:
    with patch("gravitorch.handlers.cudamem.torch.cuda.empty_cache") as empty_mock:
        EpochCudaEmptyCache().empty_cache(engine=Mock(spec=BaseEngine))
        empty_mock.assert_called_once_with()


@patch("torch.cuda.is_available", lambda *args: False)
def test_epoch_cuda_empty_cache_monitor_no_cuda() -> None:
    with patch("gravitorch.handlers.cudamem.torch.cuda.empty_cache") as empty_mock:
        EpochCudaEmptyCache().empty_cache(engine=Mock(spec=BaseEngine))
        empty_mock.assert_not_called()


#############################################
#     Tests for IterationCudaEmptyCache     #
#############################################


def test_iteration_cuda_empty_cache_str() -> None:
    assert str(IterationCudaEmptyCache()).startswith("IterationCudaEmptyCache(")


@mark.parametrize("event", EVENTS)
def test_iteration_cuda_empty_cache_event(event: str) -> None:
    assert IterationCudaEmptyCache(event)._event == event


def test_iteration_cuda_empty_cache_event_default() -> None:
    assert IterationCudaEmptyCache()._event == EngineEvents.TRAIN_ITERATION_COMPLETED


@mark.parametrize("freq", (1, 2))
def test_iteration_cuda_empty_cache_freq(freq: int) -> None:
    assert IterationCudaEmptyCache(freq=freq)._freq == freq


@mark.parametrize("freq", (0, -1))
def test_iteration_cuda_empty_cache_incorrect_freq(freq: int) -> None:
    with raises(ValueError, match="freq has to be greater than 0"):
        IterationCudaEmptyCache(freq=freq)


def test_iteration_cuda_empty_cache_freq_default() -> None:
    assert IterationCudaEmptyCache()._freq == 1


@mark.parametrize("event", EVENTS)
@mark.parametrize("freq", (1, 2))
def test_iteration_cuda_empty_cache_attach(event: str, freq: int) -> None:
    handler = IterationCudaEmptyCache(event=event, freq=freq)
    engine = Mock(spec=BaseEngine, iteration=-1, has_event_handler=Mock(return_value=False))
    handler.attach(engine)
    engine.add_event_handler.assert_called_once_with(
        event,
        ConditionalEventHandler(
            handler.empty_cache,
            condition=IterationPeriodicCondition(engine=engine, freq=freq),
            handler_kwargs={"engine": engine},
        ),
    )


def test_iteration_cuda_empty_cache_attach_duplicate() -> None:
    engine = Mock(spec=BaseEngine, iteration=-1, has_event_handler=Mock(return_value=True))
    IterationCudaEmptyCache().attach(engine)
    engine.add_event_handler.assert_not_called()


@patch("torch.cuda.is_available", lambda *args: True)
def test_iteration_cuda_empty_cache_monitor_cuda() -> None:
    with patch("gravitorch.handlers.cudamem.torch.cuda.empty_cache") as empty_mock:
        IterationCudaEmptyCache().empty_cache(engine=Mock(spec=BaseEngine))
        empty_mock.assert_called_once_with()


@patch("torch.cuda.is_available", lambda *args: False)
def test_iteration_cuda_empty_cache_monitor_no_cuda() -> None:
    with patch("gravitorch.handlers.cudamem.torch.cuda.empty_cache") as empty_mock:
        IterationCudaEmptyCache().empty_cache(engine=Mock(spec=BaseEngine))
        empty_mock.assert_not_called()
