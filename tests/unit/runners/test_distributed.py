import logging
from typing import Optional
from unittest.mock import Mock, patch

from pytest import LogCaptureFixture, mark
from torch.backends import cudnn

from gravitorch.distributed import Backend
from gravitorch.runners.distributed import (
    BaseDistributedRunner,
    BaseEngineDistributedRunner,
    resolve_dist_backend,
)

logger = logging.getLogger(__name__)


###########################################
#     Tests for BaseDistributedRunner     #
###########################################


class FakeDistributedRunner(BaseDistributedRunner):
    r"""Defines a fake distributed runner to test it because it is an abstract
    class."""

    def _run(self) -> int:
        logger.debug("This is a debug message")
        logger.info("This is an info message")
        logger.warning("This is a warning message")
        logger.error("This is an error message")
        logger.critical("This is a critical message")
        return 42


def test_base_distributed_runner():
    assert FakeDistributedRunner(dist_backend=None).run() == 42


@mark.parametrize(
    "is_main_process,log_only_main_process",
    (
        (True, True),
        (True, False),
        (False, False),
    ),
)
def test_base_distributed_runner_logging_context(
    caplog: LogCaptureFixture,
    is_main_process: bool,
    log_only_main_process: bool,
):
    with patch(
        "gravitorch.runners.distributed.dist.is_main_process", lambda *args: is_main_process
    ):
        with caplog.at_level(logging.NOTSET):
            FakeDistributedRunner(
                dist_backend=None, log_only_main_process=log_only_main_process
            ).run()
            assert len(caplog.messages) == 6


@patch("gravitorch.runners.distributed.dist.is_main_process", lambda *args: False)
def test_base_distributed_runner_log_only_main_process_true_main_process_false(
    caplog: LogCaptureFixture,
):
    with caplog.at_level(logging.NOTSET):
        FakeDistributedRunner(dist_backend=None, log_only_main_process=True).run()
        assert len(caplog.messages) == 3


##########################################
#     Tests for resolve_dist_backend     #
##########################################


@mark.parametrize("dist_backend", ("gloo", "nccl", None))
def test_resolve_dist_backend(dist_backend: Optional[str]):
    assert resolve_dist_backend(dist_backend) == dist_backend


@patch("gravitorch.distributed.utils.is_distributed_ready", lambda *args: False)
def test_resolve_dist_backend_auto_should_not_initialize():
    assert resolve_dist_backend("auto") is None


@patch("gravitorch.runners.distributed.is_distributed_ready", lambda *args: True)
@patch("gravitorch.runners.distributed.auto_dist_backend", lambda *args: Backend.GLOO)
def test_resolve_dist_backend_auto_should_initialize():
    assert resolve_dist_backend("auto") == Backend.GLOO


#################################################
#     Tests for BaseEngineDistributedRunner     #
#################################################


class FakeEngineDistributedRunner(BaseEngineDistributedRunner):
    r"""Defines a fake distributed runner to test it because it is an abstract
    class."""

    def _run(self) -> int:
        logger.debug("This is a debug message")
        logger.info("This is an info message")
        logger.warning("This is a warning message")
        logger.error("This is an error message")
        logger.critical("This is a critical message")
        return 42


def test_base_engine_distributed_runner_handlers_none():
    assert FakeEngineDistributedRunner(engine=Mock())._handlers == tuple()


def test_base_engine_distributed_runner_handlers():
    handlers = [Mock(), Mock()]
    assert FakeEngineDistributedRunner(engine=Mock(), handlers=handlers)._handlers == handlers


@mark.parametrize("random_seed", (1, 2, 3))
def test_base_engine_distributed_runner_random_seed(random_seed: int):
    assert (
        FakeEngineDistributedRunner(engine=Mock(), random_seed=random_seed)._random_seed
        == random_seed
    )


@mark.parametrize("cudnn_benchmark", (True, False))
@mark.parametrize("cudnn_deterministic", (True, False))
def test_base_engine_distributed_runner_pytorch_config(
    cudnn_benchmark: bool, cudnn_deterministic: bool
):
    FakeEngineDistributedRunner(
        engine=Mock(),
        pytorch_config={
            "cudnn_benchmark": cudnn_benchmark,
            "cudnn_deterministic": cudnn_deterministic,
        },
    ).run()
    assert cudnn.benchmark == cudnn_benchmark
    assert cudnn.deterministic == cudnn_deterministic
