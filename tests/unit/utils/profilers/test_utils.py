from objectory import OBJECT_TARGET

from gravitorch.utils.profilers import NoOpProfiler, setup_profiler

####################################
#     Tests for setup_profiler     #
####################################


def test_setup_profiler_none() -> None:
    assert isinstance(setup_profiler(None), NoOpProfiler)


def test_setup_profiler_object() -> None:
    profiler = NoOpProfiler()
    assert setup_profiler(profiler) is profiler


def test_setup_profiler_dict() -> None:
    assert isinstance(
        setup_profiler({OBJECT_TARGET: "gravitorch.utils.profilers.NoOpProfiler"}), NoOpProfiler
    )
