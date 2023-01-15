import logging
from unittest.mock import patch

from pytest import LogCaptureFixture

from gravitorch.utils.cuda_memory import (
    log_cuda_memory_summary,
    log_max_cuda_memory_allocated,
)

#############################################
#     Tests for log_cuda_memory_summary     #
#############################################


@patch("gravitorch.utils.cuda_memory.torch.cuda.is_available", lambda *args, **kwargs: True)
@patch("gravitorch.utils.cuda_memory.torch.cuda.max_memory_allocated", lambda *args, **kwargs: 12)
@patch("gravitorch.utils.cuda_memory.torch.cuda.mem_get_info", lambda *args, **kwargs: (100, 1000))
@patch("gravitorch.utils.cuda_memory.torch.cuda.memory_summary", lambda *args, **kwargs: "meow")
def test_log_cuda_memory_summary_cuda(caplog: LogCaptureFixture):
    with caplog.at_level(logging.INFO):
        log_cuda_memory_summary()
        assert len(caplog.messages) == 2


@patch("gravitorch.utils.cuda_memory.torch.cuda.is_available", lambda *args: False)
def test_log_cuda_memory_summary_no_cuda(caplog: LogCaptureFixture):
    with caplog.at_level(logging.INFO):
        log_cuda_memory_summary()
        assert len(caplog.messages) == 0


###################################################
#     Tests for log_max_cuda_memory_allocated     #
###################################################


@patch("gravitorch.utils.cuda_memory.torch.cuda.is_available", lambda *args: True)
@patch("gravitorch.utils.cuda_memory.torch.cuda.max_memory_allocated", lambda *args, **kwargs: 12)
@patch("gravitorch.utils.cuda_memory.torch.cuda.mem_get_info", lambda *args, **kwargs: (100, 1000))
def test_log_max_cuda_memory_allocated_cuda(caplog: LogCaptureFixture):
    with caplog.at_level(logging.INFO):
        log_max_cuda_memory_allocated()
        assert len(caplog.messages) == 1


@patch("gravitorch.utils.cuda_memory.torch.cuda.is_available", lambda *args: False)
def test_log_max_cuda_memory_allocated_no_cuda(caplog: LogCaptureFixture):
    with caplog.at_level(logging.INFO):
        log_max_cuda_memory_allocated()
        assert len(caplog.messages) == 0
