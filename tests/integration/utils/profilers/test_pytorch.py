from pathlib import Path

import torch

from gravitorch.utils.profilers import PyTorchProfiler

#####################################
#     Tests for PyTorchProfiler     #
#####################################


def test_pytorch_profiler_scheduled_profiler_with_tensorboard_trace(tmp_path: Path) -> None:
    trace_path = tmp_path.joinpath("trace")
    with PyTorchProfiler.scheduled_profiler_with_tensorboard_trace(
        trace_path=trace_path.as_posix(),
        wait=2,
        warmup=2,
        active=4,
    ) as profiler:
        x = torch.ones(2, 3)
        for _ in range(20):
            x += x
            profiler.step()
    assert len(tuple(trace_path.iterdir())) >= 1  # Verify at least one file has been created
