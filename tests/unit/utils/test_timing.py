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
    BatchIterTimer,
    sync_perf_counter,
    timeblock,
)

#######################################
#     Tests for sync_perf_counter     #
#######################################


@patch("gravitorch.utils.timing.torch.cuda.is_available", lambda *args: False)
def test_sync_perf_counter_no_cuda() -> None:
    with patch("gravitorch.utils.timing.torch.cuda.synchronize") as synchronize_mock:
        assert isinstance(sync_perf_counter(), float)
        synchronize_mock.assert_not_called()


@patch("gravitorch.utils.timing.torch.cuda.is_available", lambda *args: True)
def test_sync_perf_counter_cuda() -> None:
    with patch("gravitorch.utils.timing.torch.cuda.synchronize") as synchronize_mock:
        assert isinstance(sync_perf_counter(), float)
        synchronize_mock.assert_called_once_with()


###############################
#     Tests for timeblock     #
###############################


def test_timeblock(caplog: LogCaptureFixture) -> None:
    with timeblock():
        pass  # do anything
    assert len(caplog.messages) == 1
    assert caplog.messages[0].startswith("Total time: ")


def test_timeblock_custom_message(caplog: LogCaptureFixture) -> None:
    with timeblock("{time}"):
        pass  # do anything
    assert len(caplog.messages) == 1


def test_timeblock_custom_missing_time() -> None:
    with raises(ValueError, match="{time} is missing in the message"), timeblock("message"):
        pass


#######################################
#     Tests for BatchIterTimer     #
#######################################


def my_batchiter(num_batches: int = 3) -> Generator[Tensor, None, None]:
    for i in range(num_batches):
        yield torch.ones(2, 3).mul(i)


def test_batch_iter_timer_iter_get_stats() -> None:
    batchiter = BatchIterTimer(my_batchiter(), epoch=0, prefix="train")
    for batch in batchiter:
        batch += 1

    stats = batchiter.get_stats()
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


def test_batch_iter_timer_iter_get_stats_empty() -> None:
    assert BatchIterTimer(my_batchiter(), epoch=0, prefix="train").get_stats() == {}


def test_batch_iter_timer_log_time_info_without_engine(caplog: LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    batchiter = BatchIterTimer(my_batchiter(), epoch=0, prefix="train")
    for batch in batchiter:
        batch += 1

    batchiter.log_stats()
    assert len(caplog.messages) == 2


def test_batch_iter_timer_log_stats_with_engine() -> None:
    batchiter = BatchIterTimer(my_batchiter(), epoch=0, prefix="train/")
    for batch in batchiter:
        batch += 1

    engine = Mock(spec=BaseEngine)
    batchiter.log_stats(engine)
    engine.log_metrics.assert_called_once()


def test_batch_iter_timer_log_stats_empty(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        BatchIterTimer(my_batchiter(0), epoch=0, prefix="train").log_stats()
        assert len(caplog.messages) == 0
