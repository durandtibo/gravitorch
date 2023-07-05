from objectory import OBJECT_TARGET

from gravitorch.utils.profilers import NoOpProfiler, is_profiler_config, setup_profiler

########################################
#     Tests for is_profiler_config     #
########################################


def test_is_profiler_config_true() -> None:
    assert is_profiler_config({OBJECT_TARGET: "gravitorch.utils.profilers.NoOpProfiler"})


def test_is_profiler_config_false() -> None:
    assert not is_profiler_config({OBJECT_TARGET: "torch.nn.Identity"})


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
