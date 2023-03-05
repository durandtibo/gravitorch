from unittest.mock import patch

import torch
from pytest import mark

from gravitorch.utils.profilers import PyTorchProfiler

#####################################
#     Tests for PyTorchProfiler     #
#####################################


def test_pytorch_profiler_str() -> None:
    assert str(PyTorchProfiler(torch.profiler.profile())).startswith("PyTorchProfiler(")


def test_pytorch_profiler_enter_exit() -> None:
    with patch("gravitorch.utils.profilers.pytorch.torch.profiler.profile") as profile:
        with PyTorchProfiler(profile):
            profile.__enter__.assert_called_once()
    profile.__exit__.assert_called_once()


def test_pytorch_profiler_step() -> None:
    with patch("gravitorch.utils.profilers.pytorch.torch.profiler.profile") as profile:
        with PyTorchProfiler(profile) as profiler:
            profiler.step()
            profile.step.assert_called_once()


@mark.parametrize(
    "trace_path,wait,warmup,active,repeat,skip_first,record_shapes,profile_memory,with_stack,with_flops",
    (
        ("/my_path/trace", 2, 2, 4, 0, 0, False, False, False, False),
        ("/data/trace", 10, 20, 100, 30, 40, True, True, True, True),
    ),
)
def test_pytorch_profiler_scheduled_profiler_with_tensorboard_trace_parameters(
    trace_path: str,
    wait: int,
    warmup: int,
    active: int,
    repeat: int,
    skip_first: int,
    record_shapes: bool,
    profile_memory: bool,
    with_stack: bool,
    with_flops: bool,
) -> None:
    with patch(
        "gravitorch.utils.profilers.pytorch.torch.profiler.tensorboard_trace_handler"
    ) as trace_handler:
        with patch("gravitorch.utils.profilers.pytorch.torch.profiler.schedule") as schedule:
            profiler = PyTorchProfiler.scheduled_profiler_with_tensorboard_trace(
                trace_path=trace_path,
                wait=wait,
                warmup=warmup,
                active=active,
                repeat=repeat,
                skip_first=skip_first,
                record_shapes=record_shapes,
                profile_memory=profile_memory,
                with_stack=with_stack,
                with_flops=with_flops,
            )
            trace_handler.assert_called_once_with(trace_path)
            schedule.assert_called_once_with(
                wait=wait,
                warmup=warmup,
                active=active,
                repeat=repeat,
                skip_first=skip_first,
            )
            assert profiler._profiler.record_shapes == record_shapes
            assert profiler._profiler.profile_memory == profile_memory
            assert profiler._profiler.with_stack == with_stack
            assert profiler._profiler.with_flops == with_flops
