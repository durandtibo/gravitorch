from __future__ import annotations

__all__ = [
    "BATCH_LOAD_TIME_AVG_MS",
    "BATCH_LOAD_TIME_MAX_MS",
    "BATCH_LOAD_TIME_MEDIAN_MS",
    "BATCH_LOAD_TIME_MIN_MS",
    "BATCH_LOAD_TIME_PCT",
    "BATCH_LOAD_TIME_STDDEV_MS",
    "BatchLoadingTimer",
    "EPOCH_TIME_SEC",
    "ITER_TIME_AVG_MS",
    "NUM_BATCHES",
    "sync_perf_counter",
    "timeblock",
]

import logging
import time
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, TypeVar

import torch

from gravitorch.utils.exp_trackers.steps import EpochStep
from gravitorch.utils.format import human_time
from gravitorch.utils.meters import ScalarMeter

if TYPE_CHECKING:
    from gravitorch.engines import BaseEngine

logger = logging.getLogger(__name__)

T = TypeVar("T")

BATCH_LOAD_TIME_AVG_MS = "batch_load_time_avg_ms"
BATCH_LOAD_TIME_MAX_MS = "batch_load_time_max_ms"
BATCH_LOAD_TIME_MEDIAN_MS = "batch_load_time_median_ms"
BATCH_LOAD_TIME_MIN_MS = "batch_load_time_min_ms"
BATCH_LOAD_TIME_PCT = "batch_load_time_pct"
BATCH_LOAD_TIME_STDDEV_MS = "batch_load_time_stddev_ms"
EPOCH_TIME_SEC = "epoch_time_sec"
ITER_TIME_AVG_MS = "iter_time_avg_ms"
NUM_BATCHES = "num_batches"


def sync_perf_counter() -> float:
    r"""Extension of ``time.perf_counter`` that waits for all kernels in all
    streams on a CUDA device to complete.

    Returns:
        float: Same as ``time.perf_counter()``.
            See https://docs.python.org/3/library/time.html#time.perf_counter
            for more information.

    Example usage:

    .. code-block:: python

        >>> from gravitorch.utils.timing import sync_perf_counter
        >>> tic = sync_perf_counter()
        >>> x = [1, 2, 3]
        >>> toc = sync_perf_counter()
        >>> toc - tic
        7.040074708999999
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


@contextmanager
def timeblock(message: str = "Total time: {time}") -> Generator[None, None, None]:
    r"""Implements a context manager to measure the execution time of a block of
    code.

    Args:
        message (str, optional): Specifies the message displayed when
            the time is logged.

    Example usage:

    .. code-block:: python

        >>> from gravitorch.utils.timing import timeblock
        >>> with timeblock():
        ...     x = [1, 2, 3]
        INFO:gravitorch.utils.time_tracking:Total time: 0:00:00.000039
        >>> with timeblock("Training: {time}"):
        ...     x = [1, 2, 3]
        INFO:gravitorch.utils.time_tracking:Training: 0:00:00.000035
    """
    if "{time}" not in message:
        raise ValueError(f"{time} is missing in the message (received: {message})")
    start_time = sync_perf_counter()
    try:
        yield
    finally:
        logger.info(message.format(time=human_time(sync_perf_counter() - start_time)))


class BatchLoadingTimer(Iterable[T]):
    r"""Implements an iterator around a batch loader iterable to monitor the
    time performances.

    Args:
        batch_loader (Iterable): Specifies the batch loader to measure
            the time performances.
        epoch (int): Specifies the epoch.
        prefix (str): Specifies the prefix used to log the values.


    Example usage:

    .. code-block:: python

        >>> from gravitorch.utils.timing import BatchLoadingTimer
        >>> my_batch_loader = [1, 2, 3, 4, 5]
        >>> batch_loader = BatchLoadingTimer(my_batch_loader, epoch=0, prefix="train")
        >>> for batch in batch_loader:
        ...     pass # do something
        >>> batch_loader.get_stats()
        {'batch_load_time_avg_ms': 0.003508200001078876,
         'batch_load_time_max_ms': 0.010916999997334642,
         'batch_load_time_median_ms': 0.0016660000028423383,
         'batch_load_time_min_ms': 0.0015419999996879596,
         'batch_load_time_pct': 36.012564686438395,
         'batch_load_time_stddev_ms': 0.004143146270507714,
         'iter_time_avg_ms': 0.00974159999884705,
         'epoch_time_sec': 4.870799999423525e-05,
         'num_batches': 5}
        >>> batch_loader.log_stats()
        INFO:gravitorch.utils.timing:Epoch: 0 | batch: 5 | total time: 0.000 s | avg iter time: 0.010 ms | avg batch load time: 0.004 ms | avg batch load pct: 36.01 %  # noqa: E501,B950
        INFO:gravitorch.utils.timing:Batch loading time min/avg/median/max/stddev = 0.002/0.004/0.002/0.011/0.004 ms  # noqa: E501,B950
    """

    def __init__(self, batch_loader: Iterable[T], epoch: int, prefix: str):
        self._batch_loader = batch_loader
        self._epoch = epoch
        self._prefix = prefix
        self._meter = ScalarMeter(max_size=100)

        self._start_time: float | None = None
        self._end_time: float | None = None

    def __iter__(self):
        self._start_time = sync_perf_counter()
        tic = self._start_time
        for batch in self._batch_loader:
            self._meter.update(sync_perf_counter() - tic)
            yield batch
            tic = sync_perf_counter()
        self._end_time = sync_perf_counter()

    def get_stats(self) -> dict[str, Any]:
        r"""Gets some stats about the batch loading performances.

        The computed stats are:

            - ``'batch_load_time_avg_ms'``: The average time in
                milliseconds to load a batch.
            - ``'batch_load_time_max_ms'``: The maximum time in
                milliseconds to load a batch.
            - ``'batch_load_time_median_ms'``: The median time in
                milliseconds to load a batch.
            - ``'batch_load_time_min_ms'``: The minimum time in
                milliseconds to load a batch.
            - ``'batch_load_time_stddev_ms'``: The standard deviation
                time in milliseconds to load a batch.
            - ``'num_batches'``: The number of batches.
            - ``'epoch_time_sec'``: The total time in seconds to
                iterate and process all the batches. It corresponds
                to the time to computed one epoch.
            - ``'iter_time_avg_ms'``: The average time in
                milliseconds to iterate and process a batch. It
                corresponds to the time to load and process a batch.
            - ``'batch_load_time_pct'``: The percentage of time spent
                to load the batches. The lower the better. This metric
                is important to know if the batch loading is a
                bottleneck or not.

        This method has to be called after the batches have been
        iterated.

        Returns:
            dict: The dictionary with the batch loading stats. The
                dictionary is empty if there is no batches.
        """
        if self._meter.count == 0:
            return {}
        epoch_time_sec = self._end_time - self._start_time
        iter_time_avg_ms = 1000 * epoch_time_sec / self._meter.count
        batch_load_avg_ms = 1000 * self._meter.average()
        return {
            BATCH_LOAD_TIME_AVG_MS: batch_load_avg_ms,
            BATCH_LOAD_TIME_MAX_MS: 1000 * self._meter.max(),
            BATCH_LOAD_TIME_MEDIAN_MS: 1000 * self._meter.median(),
            BATCH_LOAD_TIME_MIN_MS: 1000 * self._meter.min(),
            BATCH_LOAD_TIME_PCT: 100 * batch_load_avg_ms / iter_time_avg_ms,
            BATCH_LOAD_TIME_STDDEV_MS: 1000 * self._meter.std(),
            ITER_TIME_AVG_MS: iter_time_avg_ms,
            EPOCH_TIME_SEC: epoch_time_sec,
            NUM_BATCHES: int(self._meter.count),
        }

    def log_stats(self, engine: BaseEngine | None = None) -> None:
        r"""Logs the time statistics about the batch loading.

        This method has to be called after the batches have been
        iterated.

        Args:
            engine (``BaseEngine`` or None): Specifies an engine if
                you want to log the time metrics in the engine. If
                ``None``, the time metrics are not logged in an
                engine. In each case, the time metrics are logged in
                the python logger. Default: ``None``
        """
        if self._meter.count == 0:
            return  # Do nothing if there is no mini-batch.
        stats = self.get_stats()
        epoch_time_sec = stats[EPOCH_TIME_SEC]
        epoch_time_str = (
            human_time(epoch_time_sec) if epoch_time_sec > 10 else f"{epoch_time_sec:.3f} s"
        )
        logger.info(
            f"Epoch: {self._epoch:,} | batch: {stats[NUM_BATCHES]:,} | "
            f"total time: {epoch_time_str} | "
            f"avg iter time: {stats[ITER_TIME_AVG_MS]:.3f} ms | "
            f"avg batch load time: {stats[BATCH_LOAD_TIME_AVG_MS]:,.3f} ms | "
            f"avg batch load pct: {stats[BATCH_LOAD_TIME_PCT]:.2f} %"
        )
        logger.info(
            f"Batch loading time min/avg/median/max/stddev = {stats[BATCH_LOAD_TIME_MIN_MS]:.3f}/"
            f"{stats[BATCH_LOAD_TIME_AVG_MS]:.3f}/{stats[BATCH_LOAD_TIME_MEDIAN_MS]:.3f}/"
            f"{stats[BATCH_LOAD_TIME_MAX_MS]:.3f}/{stats[BATCH_LOAD_TIME_STDDEV_MS]:.3f} ms"
        )
        if engine:
            engine.log_metrics(
                {f"{self._prefix}{key}": value for key, value in stats.items()},
                step=EpochStep(self._epoch),
            )
