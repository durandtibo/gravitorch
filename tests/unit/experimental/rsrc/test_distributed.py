import logging

from pytest import LogCaptureFixture, raises

from gravitorch.distributed import UnknownBackendError
from gravitorch.distributed import backend as dist_backend
from gravitorch.experimental.rsrc.distributed import DistributedContext

########################################
#     Tests for DistributedContext     #
########################################


def test_distributed_context_str():
    assert str(DistributedContext(backend=None)).startswith("DistributedContext(")


def test_distributed_context():
    with DistributedContext(backend=None):
        assert dist_backend() is None


def test_distributed_context_incorrect_backend():
    with raises(UnknownBackendError):
        with DistributedContext(backend="incorrect"):
            pass


def test_distributed_context_log_info_true(caplog: LogCaptureFixture):
    with caplog.at_level(logging.INFO):
        with DistributedContext(backend=None, log_info=True):
            pass
        assert len(caplog.messages) == 3


def test_distributed_context_log_info_false(caplog: LogCaptureFixture):
    with caplog.at_level(logging.INFO):
        with DistributedContext(backend=None):
            pass
        assert len(caplog.messages) == 2
