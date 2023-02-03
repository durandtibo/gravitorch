import logging
from typing import Union
from unittest.mock import patch

from pytest import mark

from gravitorch.rsrc.logging import Logging, LoggingState

##################################
#     Tests for LoggingState     #
##################################


def test_logging_state_create():
    state = LoggingState.create()
    assert isinstance(state, LoggingState)
    assert isinstance(state.disabled_level, int)


def test_logging_state_restore():
    with Logging():
        LoggingState(disabled_level=42).restore()
        assert logging.root.manager.disable == 42


#############################
#     Tests for Logging     #
#############################


def test_logging_str():
    assert str(Logging()).startswith("Logging(")


@mark.parametrize("disabled_level", ("WARNING", 30))
def test_logging_disabled_level(disabled_level: Union[int, str]):
    assert Logging(disabled_level=disabled_level)._disabled_level == 30


@mark.parametrize("disabled_level", (15, 42))
@mark.parametrize("only_main_process", (True, False))
def test_logging_main_process(disabled_level: int, only_main_process: bool):
    default = logging.root.manager.disable
    with patch("gravitorch.distributed.comm.is_main_process", lambda *args: True):
        with Logging(only_main_process=only_main_process, disabled_level=disabled_level):
            assert logging.root.manager.disable == default
    assert logging.root.manager.disable == default


@mark.parametrize("disabled_level", (15, 42))
def test_logging_non_main_process_only_main_process_true(disabled_level: int):
    default = logging.root.manager.disable
    with patch("gravitorch.distributed.comm.is_main_process", lambda *args: False):
        with Logging(only_main_process=True, disabled_level=disabled_level):
            assert logging.root.manager.disable == disabled_level
    assert logging.root.manager.disable == default


@mark.parametrize("disabled_level", (15, 42))
def test_logging_non_main_process_only_main_process_false(disabled_level: int):
    default = logging.root.manager.disable
    with patch("gravitorch.distributed.comm.is_main_process", lambda *args: False):
        with Logging(only_main_process=False, disabled_level=disabled_level):
            assert logging.root.manager.disable == default
    assert logging.root.manager.disable == default


def test_logging_reentrant():
    default = logging.root.manager.disable
    resource = Logging()
    with resource:
        with resource:
            assert logging.root.manager.disable == default
    assert logging.root.manager.disable == default
