import logging
from collections.abc import Generator
from unittest.mock import Mock, patch

import torch
from pytest import LogCaptureFixture, raises
from torch import Tensor

from gravitorch.engines import BaseEngine
from gravitorch.utils.timing import (
    BATCH_LOAD_TIME_AVG_MS,
    BATCH_LOAD_TIME_MAX_MS,
    BATCH_LOAD_TIME_MEDIAN_MS,
    BATCH_LOAD_TIME_MIN_MS,
    BATCH_LOAD_TIME_PCT,
    BATCH_LOAD_TIME_STDDEV_MS,
    EPOCH_TIME_SEC,
    ITER_TIME_AVG_MS,
    NUM_BATCHES,
    BatchLoadingTimer,
    sync_perf_counter,
    timeblock,
)

#######################################
#     Tests for sync_perf_counter     #
#######################################


@patch("gravitorch.utils.timing.torch.cuda.is_available", lambda *args: False)
def test_sync_perf_counter_no_cuda():
    with patch("gravitorch.utils.timing.torch.cuda.synchronize") as synchronize_mock:
        assert isinstance(sync_perf_counter(), float)
        synchronize_mock.assert_not_called()


@patch("gravitorch.utils.timing.torch.cuda.is_available", lambda *args: True)
def test_sync_perf_counter_cuda():
    with patch("gravitorch.utils.timing.torch.cuda.synchronize") as synchronize_mock:
        assert isinstance(sync_perf_counter(), float)
        synchronize_mock.assert_called_once_with()


###############################
#     Tests for timeblock     #
###############################


def test_timeblock(caplog: LogCaptureFixture):
    with timeblock():
        pass  # do anything
    assert len(caplog.messages) == 1
    assert caplog.messages[0].startswith("Total time: ")


def test_timeblock_custom_message(caplog: LogCaptureFixture):
    with timeblock("{time}"):
        pass  # do anything
    assert len(caplog.messages) == 1


def test_timeblock_custom_missing_time():
    with raises(ValueError):
        with timeblock("message"):
            pass


#######################################
#     Tests for BatchLoadingTimer     #
#######################################


def my_batch_loader(num_batches: int = 3) -> Generator[Tensor, None, None]:
    for i in range(num_batches):
        yield torch.ones(2, 3).mul(i)


def test_batch_loading_timer_iter_get_stats():
    batch_loader = BatchLoadingTimer(my_batch_loader(), epoch=0, prefix="train")
    for batch in batch_loader:
        batch += 1

    stats = batch_loader.get_stats()
    assert len(stats) == 9
    assert isinstance(stats[BATCH_LOAD_TIME_AVG_MS], float)
    assert isinstance(stats[BATCH_LOAD_TIME_MAX_MS], float)
    assert isinstance(stats[BATCH_LOAD_TIME_MEDIAN_MS], float)
    assert isinstance(stats[BATCH_LOAD_TIME_MIN_MS], float)
    assert isinstance(stats[BATCH_LOAD_TIME_PCT], float)
    assert isinstance(stats[BATCH_LOAD_TIME_STDDEV_MS], float)
    assert isinstance(stats[EPOCH_TIME_SEC], float)
    assert isinstance(stats[ITER_TIME_AVG_MS], float)
    assert stats[NUM_BATCHES] == 3


def test_batch_loading_timer_iter_get_stats_empty():
    assert BatchLoadingTimer(my_batch_loader(), epoch=0, prefix="train").get_stats() == {}


def test_batch_loading_timer_log_time_info_without_engine(caplog: LogCaptureFixture):
    caplog.set_level(logging.INFO)
    batch_loader = BatchLoadingTimer(my_batch_loader(), epoch=0, prefix="train")
    for batch in batch_loader:
        batch += 1

    batch_loader.log_stats()
    assert len(caplog.messages) == 2


def test_batch_loading_timer_log_stats_with_engine():
    batch_loader = BatchLoadingTimer(my_batch_loader(), epoch=0, prefix="train/")
    for batch in batch_loader:
        batch += 1

    engine = Mock(spec=BaseEngine)
    batch_loader.log_stats(engine)
    engine.log_metrics.assert_called_once()


def test_batch_loading_timer_log_stats_empty(caplog: LogCaptureFixture):
    with caplog.at_level(logging.INFO):
        BatchLoadingTimer(my_batch_loader(0), epoch=0, prefix="train").log_stats()
        assert len(caplog.messages) == 0
