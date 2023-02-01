import logging

from pytest import LogCaptureFixture

from gravitorch.experimental.rsrc import LogSysInfo

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
