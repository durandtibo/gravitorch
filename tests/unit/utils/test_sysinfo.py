import logging
from unittest.mock import patch

from pytest import LogCaptureFixture

from gravitorch.testing import psutil_available
from gravitorch.utils.sysinfo import (
    cpu_human_summary,
    log_system_info,
    swap_memory_human_summary,
    virtual_memory_human_summary,
)

#######################################
#     Tests for cpu_human_summary     #
#######################################


@psutil_available
def test_cpu_human_summary() -> None:
    assert cpu_human_summary().startswith("CPU")


def test_cpu_human_summary_no_psutil(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        with patch("gravitorch.utils.sysinfo.is_psutil_available", lambda *args: False):
            assert cpu_human_summary() == "CPU - N/A"
            assert len(caplog.messages) == 1


#####################################
#     Tests for log_system_info     #
#####################################


@psutil_available
def test_log_system_info(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        log_system_info()
    assert len(caplog.messages) == 3


def test_log_system_info_no_psutil(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        with patch("gravitorch.utils.sysinfo.is_psutil_available", lambda *args: False):
            log_system_info()
            assert len(caplog.messages) == 3


###############################################
#     Tests for swap_memory_human_summary     #
###############################################


@psutil_available
def test_swap_memory_human_summary() -> None:
    assert swap_memory_human_summary().startswith("swap memory")


def test_swap_memory_human_summary_no_psutil(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        with patch("gravitorch.utils.sysinfo.is_psutil_available", lambda *args: False):
            assert swap_memory_human_summary() == "swap memory - N/A"
            assert len(caplog.messages) == 1


##################################################
#     Tests for virtual_memory_human_summary     #
##################################################


@psutil_available
def test_virtual_memory_human_summary() -> None:
    assert virtual_memory_human_summary().startswith("virtual memory")


def test_virtual_memory_human_summary_no_psutil(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        with patch("gravitorch.utils.sysinfo.is_psutil_available", lambda *args: False):
            assert virtual_memory_human_summary() == "virtual memory - N/A"
            assert len(caplog.messages) == 1
