import logging

from pytest import LogCaptureFixture

from gravitorch.rsrc import LogCudaMemory, LogSysInfo
from gravitorch.testing import psutil_available

###################################
#     Tests for LogCudaMemory     #
###################################


def test_log_cuda_memory_str() -> None:
    assert str(LogCudaMemory()).startswith("LogCudaMemory(")


def test_log_cuda_memory(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        with LogCudaMemory():
            pass
        assert 0 <= len(caplog.messages) <= 2


################################
#     Tests for LogSysInfo     #
################################


def test_log_sys_info_str() -> None:
    assert str(LogSysInfo()).startswith("LogSysInfo(")


@psutil_available
def test_log_sys_info(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        with LogSysInfo():
            pass
        assert len(caplog.messages) == 6
