from torch.backends import cuda

from gravitorch.experimental.rsrc import PyTorchCudaBackend
from gravitorch.runners import BaseResourceRunner

########################################
#     Tests for BaseResourceRunner     #
########################################


class FakeResourceRunner(BaseResourceRunner):
    r"""Defines a fake runner to test ``BaseResourceRunner`` because it has an
    abstract method."""

    def _run(self) -> int:
        return 42


def test_base_resource_runner_run_without_resources():
    assert FakeResourceRunner().run() == 42


def test_base_resource_runner_run_with_resources():
    default = cuda.matmul.allow_tf32
    assert FakeResourceRunner(resources=(PyTorchCudaBackend(allow_tf32=True),)).run() == 42
    assert cuda.matmul.allow_tf32 == default
