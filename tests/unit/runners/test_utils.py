import logging

from objectory import OBJECT_TARGET
from pytest import LogCaptureFixture, mark
from torch.backends import cudnn

from gravitorch.runners import TrainingRunner
from gravitorch.runners.utils import (
    configure_pytorch,
    setup_runner,
    show_cuda_info,
    show_cudnn_info,
)

##################################
#     Tests for setup_runner     #
##################################


def test_setup_runner_object() -> None:
    runner = TrainingRunner(engine={})
    assert setup_runner(runner) is runner


def test_setup_runner_dict() -> None:
    assert isinstance(
        setup_runner({OBJECT_TARGET: "gravitorch.runners.TrainingRunner", "engine": {}}),
        TrainingRunner,
    )


#######################################
#     Tests for configure_pytorch     #
#######################################


@mark.parametrize("cudnn_benchmark", (True, False))
def test_configure_pytorch_cudnn_benchmark(cudnn_benchmark: bool) -> None:
    configure_pytorch(cudnn_benchmark=cudnn_benchmark)
    assert cudnn.benchmark == cudnn_benchmark


@mark.parametrize("cudnn_deterministic", (True, False))
def test_configure_pytorch_cudnn_deterministic(cudnn_deterministic: bool) -> None:
    configure_pytorch(cudnn_deterministic=cudnn_deterministic)
    assert cudnn.deterministic == cudnn_deterministic


#####################################
#     Tests for show_cuda_info     #
#####################################


def test_show_cuda_info(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        show_cuda_info()
        assert len(caplog.messages[0]) > 0  # The message should not be empty


#####################################
#     Tests for show_cudnn_info     #
#####################################


def test_show_cudnn_info(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        show_cudnn_info()
        assert len(caplog.messages[0]) > 0  # The message should not be empty
