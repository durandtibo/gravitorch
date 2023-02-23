import logging

from pytest import LogCaptureFixture

from gravitorch.testing import cuda_available
from gravitorch.utils.cudamem import (
    log_cuda_memory_summary,
    log_max_cuda_memory_allocated,
)

#############################################
#     Tests for log_cuda_memory_summary     #
#############################################


@cuda_available
def test_log_cuda_memory_summary_cuda(caplog: LogCaptureFixture):
    with caplog.at_level(logging.INFO):
        log_cuda_memory_summary()
        assert len(caplog.messages) == 2


###################################################
#     Tests for log_max_cuda_memory_allocated     #
###################################################


@cuda_available
def test_log_max_cuda_memory_allocated_cuda(caplog: LogCaptureFixture):
    with caplog.at_level(logging.INFO):
        log_max_cuda_memory_allocated()
        assert len(caplog.messages) == 1
