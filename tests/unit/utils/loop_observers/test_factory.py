from pathlib import Path

from objectory import OBJECT_TARGET

from gravitorch.utils.loop_observers import (
    NoOpLoopObserver,
    PyTorchBatchSaver,
    setup_loop_observer,
)

#########################################
#     Tests for setup_loop_observer     #
#########################################


def test_setup_loop_observer_none():
    assert isinstance(setup_loop_observer(None), NoOpLoopObserver)


def test_setup_loop_observer_object():
    loop_observer = NoOpLoopObserver()
    assert setup_loop_observer(loop_observer) is loop_observer


def test_setup_loop_observer_dict(tmp_path: Path):
    assert isinstance(
        setup_loop_observer(
            {
                OBJECT_TARGET: "gravitorch.utils.loop_observers.PyTorchBatchSaver",
                "path": tmp_path.joinpath("batch.pt"),
            }
        ),
        PyTorchBatchSaver,
    )
