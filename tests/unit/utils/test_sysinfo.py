import logging
from unittest.mock import patch

from pytest import LogCaptureFixture, raises

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


def test_cpu_human_summary_no_psutil() -> None:
    with patch("gravitorch.utils.imports.is_psutil_available", lambda *args: False):
        with raises(RuntimeError, match="`psutil` package is required but not installed."):
            cpu_human_summary()


#####################################
#     Tests for log_system_info     #
#####################################


@psutil_available
def test_log_system_info(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        log_system_info()
    assert len(caplog.messages) == 3


def test_log_system_info_no_psutil() -> None:
    with patch("gravitorch.utils.imports.is_psutil_available", lambda *args: False):
        with raises(RuntimeError, match="`psutil` package is required but not installed."):
            log_system_info()


###############################################
#     Tests for swap_memory_human_summary     #
###############################################


@psutil_available
def test_swap_memory_human_summary() -> None:
    assert swap_memory_human_summary().startswith("swap memory")


def test_swap_memory_human_summary_no_psutil() -> None:
    with patch("gravitorch.utils.imports.is_psutil_available", lambda *args: False):
        with raises(RuntimeError, match="`psutil` package is required but not installed."):
            swap_memory_human_summary()


##################################################
#     Tests for virtual_memory_human_summary     #
##################################################


@psutil_available
def test_virtual_memory_human_summary() -> None:
    assert virtual_memory_human_summary().startswith("virtual memory")


def test_virtual_memory_human_summary_no_psutil() -> None:
    with patch("gravitorch.utils.imports.is_psutil_available", lambda *args: False):
        with raises(RuntimeError, match="`psutil` package is required but not installed."):
            virtual_memory_human_summary()
