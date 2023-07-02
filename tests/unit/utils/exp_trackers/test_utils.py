from unittest.mock import Mock, patch

import torch
from objectory import OBJECT_TARGET

from gravitorch.utils.exp_trackers import (
    BaseExpTracker,
    NoOpExpTracker,
    is_exp_tracker_config,
    main_process_only,
    sanitize_metrics,
    setup_exp_tracker,
)

###########################################
#     Tests for is_exp_tracker_config     #
###########################################


def test_is_exp_tracker_config_true() -> None:
    assert is_exp_tracker_config({"_target_": "gravitorch.utils.exp_trackers.NoOpExpTracker"})


def test_is_exp_tracker_config_false() -> None:
    assert not is_exp_tracker_config({"_target_": "torch.nn.Identity"})


#######################################
#     Tests for setup_exp_tracker     #
#######################################


def test_setup_exp_tracker_none() -> None:
    assert isinstance(setup_exp_tracker(None), NoOpExpTracker)


def test_setup_exp_tracker_config() -> None:
    assert isinstance(
        setup_exp_tracker({OBJECT_TARGET: "gravitorch.utils.exp_trackers.NoOpExpTracker"}),
        NoOpExpTracker,
    )


def test_setup_exp_tracker_object() -> None:
    exp_tracker = NoOpExpTracker()
    assert setup_exp_tracker(exp_tracker) is exp_tracker


######################################
#     Tests for sanitize_metrics     #
######################################


def test_sanitize_metrics() -> None:
    assert sanitize_metrics(
        {
            "bool": True,
            "int": 1,
            "float": 2.5,
            "str": "abc",
            "tensor": torch.ones(1),
            "none": None,
        }
    ) == {
        "bool": True,
        "int": 1,
        "float": 2.5,
    }


def test_sanitize_metrics_empty() -> None:
    assert sanitize_metrics({}) == {}


#######################################
#     Tests for setup_exp_tracker     #
#######################################


@patch("gravitorch.utils.exp_trackers.utils.dist.is_main_process", lambda *args: True)
def test_main_process_only_main_process() -> None:
    tracker = Mock(sepc=BaseExpTracker)
    assert main_process_only(tracker) is tracker


@patch("gravitorch.utils.exp_trackers.utils.dist.is_main_process", lambda *args: False)
def test_main_process_only_non_main_process() -> None:
    assert isinstance(
        main_process_only({OBJECT_TARGET: "gravitorch.utils.exp_trackers.TensorBoardExpTracker"}),
        NoOpExpTracker,
    )
