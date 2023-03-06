import logging

from pytest import LogCaptureFixture

from gravitorch.utils.sysinfo import (
    cpu_human_summary,
    log_system_info,
    swap_memory_human_summary,
    virtual_memory_human_summary,
)

#######################################
#     Tests for cpu_human_summary     #
#######################################


def test_cpu_human_summary() -> None:
    assert cpu_human_summary().startswith("CPU")


#####################################
#     Tests for log_system_info     #
#####################################


def test_log_system_info(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        log_system_info()
    assert len(caplog.messages) == 3


###############################################
#     Tests for swap_memory_human_summary     #
###############################################


def test_swap_memory_human_summary() -> None:
    assert swap_memory_human_summary().startswith("swap memory")


##################################################
#     Tests for virtual_memory_human_summary     #
##################################################


def test_virtual_memory_human_summary() -> None:
    assert virtual_memory_human_summary().startswith("virtual memory")
