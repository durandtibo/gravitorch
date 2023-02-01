import logging

from pytest import LogCaptureFixture

from gravitorch.experimental.rsrc import LogCudaMemory, LogSysInfo

###################################
#     Tests for LogCudaMemory     #
###################################


def test_log_cuda_memory_str():
    assert str(LogCudaMemory()).startswith("LogCudaMemory(")


def test_log_cuda_memory(caplog: LogCaptureFixture):
    with caplog.at_level(logging.INFO):
        with LogCudaMemory():
            pass
        assert 0 <= len(caplog.messages) <= 2


################################
#     Tests for LogSysInfo     #
################################


def test_log_sys_info_str():
    assert str(LogSysInfo()).startswith("LogSysInfo(")


def test_log_sys_info(caplog: LogCaptureFixture):
    with caplog.at_level(logging.INFO):
        with LogSysInfo():
            pass
        assert len(caplog.messages) == 6
