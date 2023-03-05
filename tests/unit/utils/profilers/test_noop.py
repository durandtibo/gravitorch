from gravitorch.utils.profilers import NoOpProfiler

##################################
#     Tests for NoOpProfiler     #
##################################


def test_noop_profiler_str() -> None:
    assert str(NoOpProfiler()) == "NoOpProfiler()"


def test_noop_profiler_step() -> None:
    with NoOpProfiler() as profiler:
        profiler.step()
