from gravitorch.utils.profilers import NoOpProfiler

##################################
#     Tests for NoOpProfiler     #
##################################


def test_noop_profiler_str():
    assert str(NoOpProfiler()) == "NoOpProfiler()"


def test_noop_profiler_step():
    with NoOpProfiler() as profiler:
        profiler.step()
