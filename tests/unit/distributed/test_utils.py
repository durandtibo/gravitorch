import logging
import os
from unittest.mock import patch

from pytest import LogCaptureFixture

from gravitorch import distributed as dist
from gravitorch.distributed.utils import (
    has_slurm_distributed_env_vars,
    has_torch_distributed_env_vars,
    is_distributed_ready,
    is_slurm_job,
    show_all_slurm_env_vars,
    show_distributed_context_info,
    show_distributed_env_vars,
    show_slurm_env_vars,
    show_torch_distributed_env_vars,
)

##################################
#     Tests for is_slurm_job     #
##################################


@patch.dict(os.environ, {}, clear=True)
def test_is_slurm_job_false() -> None:
    assert not is_slurm_job()


@patch.dict(os.environ, {"SLURM_JOB_ID": "12345"}, clear=True)
def test_is_slurm_job_true() -> None:
    assert is_slurm_job()


##########################################
#     Tests for is_distributed_ready     #
##########################################


@patch.dict(os.environ, {}, clear=True)
def test_is_distributed_ready_false() -> None:
    assert not is_distributed_ready()


@patch.dict(
    os.environ,
    {
        dist.GROUP_RANK: "0",
        dist.LOCAL_RANK: "0",
        dist.LOCAL_WORLD_SIZE: "1",
        dist.MASTER_ADDR: "localhost",
        dist.MASTER_PORT: "12345",
        dist.RANK: "0",
        dist.ROLE_RANK: "0",
        dist.ROLE_WORLD_SIZE: "1",
        dist.TORCHELASTIC_MAX_RESTARTS: "1",
        dist.TORCHELASTIC_RESTART_COUNT: "0",
        dist.TORCHELASTIC_RUN_ID: "12345",
        dist.WORLD_SIZE: "1",
    },
    clear=True,
)
def test_is_distributed_ready_true() -> None:
    assert is_distributed_ready()


@patch.dict(os.environ, {dist.SLURM_JOB_ID: "123"}, clear=True)
def test_is_distributed_ready_slurm_false() -> None:
    assert not is_distributed_ready()


@patch.dict(
    os.environ,
    {
        dist.SLURM_JOB_ID: "123",
        dist.SLURM_PROCID: "0",
        dist.SLURM_LOCALID: "0",
        dist.SLURM_NTASKS: "1",
        dist.SLURM_JOB_NODELIST: "something",
    },
    clear=True,
)
def test_is_distributed_ready_slurm_false_ntask_1() -> None:
    assert not is_distributed_ready()


@patch.dict(
    os.environ,
    {
        dist.SLURM_JOB_ID: "123",
        dist.SLURM_PROCID: "3",
        dist.SLURM_LOCALID: "1",
        dist.SLURM_NTASKS: "4",
        dist.SLURM_JOB_NODELIST: "something",
    },
    clear=True,
)
def test_is_distributed_ready_slurm_true() -> None:
    assert is_distributed_ready()


####################################################
#     Tests for has_slurm_distributed_env_vars     #
####################################################


@patch.dict(os.environ, {}, clear=True)
def test_has_slurm_distributed_env_vars_false() -> None:
    assert not has_slurm_distributed_env_vars()


@patch.dict(
    os.environ,
    {
        dist.SLURM_JOB_ID: "123",
        dist.SLURM_PROCID: "3",
        dist.SLURM_LOCALID: "1",
        dist.SLURM_JOB_NODELIST: "4",
    },
    clear=True,
)
def test_has_slurm_distributed_env_vars_false_because_ntask_missing() -> None:
    assert not has_slurm_distributed_env_vars()


@patch.dict(
    os.environ,
    {
        dist.SLURM_JOB_ID: "123",
        dist.SLURM_PROCID: "3",
        dist.SLURM_LOCALID: "1",
        dist.SLURM_NTASKS: "4",
        dist.SLURM_JOB_NODELIST: "something",
    },
    clear=True,
)
def test_has_slurm_distributed_env_vars_true() -> None:
    assert has_slurm_distributed_env_vars()


####################################################
#     Tests for has_torch_distributed_env_vars     #
####################################################


@patch.dict(os.environ, {}, clear=True)
def test_has_torch_distributed_env_vars_false() -> None:
    assert not has_torch_distributed_env_vars()


@patch.dict(
    os.environ,
    {
        dist.GROUP_RANK: "0",
        dist.LOCAL_RANK: "0",
        dist.LOCAL_WORLD_SIZE: "1",
        dist.MASTER_ADDR: "localhost",
        dist.MASTER_PORT: "12345",
        dist.ROLE_RANK: "0",
        dist.ROLE_WORLD_SIZE: "1",
        dist.TORCHELASTIC_MAX_RESTARTS: "1",
        dist.TORCHELASTIC_RESTART_COUNT: "0",
        dist.TORCHELASTIC_RUN_ID: "12345",
        dist.WORLD_SIZE: "1",
    },
    clear=True,
)
def test_has_torch_distributed_env_vars_false_because_rank_missing() -> None:
    assert not has_torch_distributed_env_vars()


@patch.dict(
    os.environ,
    {
        dist.GROUP_RANK: "0",
        dist.LOCAL_RANK: "0",
        dist.LOCAL_WORLD_SIZE: "1",
        dist.MASTER_ADDR: "localhost",
        dist.MASTER_PORT: "12345",
        dist.RANK: "0",
        dist.ROLE_RANK: "0",
        dist.ROLE_WORLD_SIZE: "1",
        dist.TORCHELASTIC_MAX_RESTARTS: "1",
        dist.TORCHELASTIC_RESTART_COUNT: "0",
        dist.TORCHELASTIC_RUN_ID: "12345",
        dist.WORLD_SIZE: "1",
    },
    clear=True,
)
def test_has_torch_distributed_env_vars_true() -> None:
    assert has_torch_distributed_env_vars()


###############################################
#     Tests for show_distributed_env_vars     #
###############################################


def test_show_distributed_env_vars(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        show_distributed_env_vars()
        assert len(caplog.messages[0]) > 0  # The message should not be empty


@patch.dict(os.environ, {"SLURM_JOB_ID": "12345"}, clear=True)
def test_show_distributed_env_vars_slurm_job(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        show_distributed_env_vars()
        assert len(caplog.messages[0]) > 0  # The message should not be empty


#####################################################
#     Tests for show_torch_distributed_env_vars     #
#####################################################


@patch.dict(os.environ, {}, clear=True)
def test_show_torch_distributed_env_vars_missing(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        show_torch_distributed_env_vars()
        assert len(caplog.messages[0]) > 0  # The message should not be empty


@patch.dict(os.environ, {dist.WORLD_SIZE: "4", dist.RANK: "1"}, clear=True)
def test_show_torch_distributed_env_vars_partial(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        show_torch_distributed_env_vars()
        assert len(caplog.messages[0]) > 0  # The message should not be empty


#############################################
#     Tests for show_all_slurm_env_vars     #
#############################################


@patch.dict(os.environ, {}, clear=True)
def test_show_all_slurm_env_vars_non_slurm_job(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        show_all_slurm_env_vars()
        assert len(caplog.messages[0]) > 0  # The message should not be empty


@patch.dict(os.environ, {"SLURM_JOB_ID": "12345"}, clear=True)
def test_show_all_slurm_env_vars_slurm_job(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        show_all_slurm_env_vars()
        assert len(caplog.messages[0]) > 0  # The message should not be empty


#########################################
#     Tests for show_slurm_env_vars     #
#########################################


@patch.dict(os.environ, {}, clear=True)
def test_show_slurm_env_vars_non_slurm_job(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        show_slurm_env_vars()
        assert len(caplog.messages[0].split("\n")) == 7


@patch.dict(os.environ, {"SLURM_JOB_ID": "12345"}, clear=True)
def test_show_slurm_env_vars_slurm_job(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        show_slurm_env_vars()
        assert len(caplog.messages[0].split("\n")) == 7


###################################################
#     Tests for show_distributed_context_info     #
###################################################


def test_show_distributed_context_info(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        show_distributed_context_info()
        assert len(caplog.messages[0]) > 0  # The message should not be empty
