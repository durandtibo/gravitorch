import logging
from unittest.mock import Mock

import numpy as np
import torch
from pytest import LogCaptureFixture

from gravitorch.engines import BaseEngine
from gravitorch.utils.exp_trackers import EpochStep
from gravitorch.utils.metric_tracker import ScalarMetricTracker

#########################################
#     Tests for ScalarMetricTracker     #
#########################################


def test_scalar_metric_tracker_update_multiple_values() -> None:
    tracker = ScalarMetricTracker()
    tracker.update(
        {
            "int": 42,
            "float": 3.5,
            "tensor_int": torch.tensor(10),
            "tensor_float": torch.tensor(10.1),
            "tensor_dim_1": torch.tensor([11]),
            "tensor_dim_2": torch.tensor([[12]]),
            "big_tensor": torch.ones(2, 3),
            "numpy": np.array([3]),
        }
    )
    assert len(tracker._metrics) == 6
    assert "int" in tracker._metrics
    assert "float" in tracker._metrics
    assert "tensor_int" in tracker._metrics
    assert "tensor_float" in tracker._metrics
    assert "tensor_dim_1" in tracker._metrics
    assert "tensor_dim_2" in tracker._metrics


def test_scalar_metric_tracker_log_average_value(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        tracker = ScalarMetricTracker()
        tracker.update({"metric0": 3, "metric1": 12})
        tracker.update({"metric0": 1, "metric1": 10})
        tracker.log_average_value()
        assert caplog.messages[0] == "metric0: 2.000000"
        assert caplog.messages[1] == "metric1: 11.000000"


def test_scalar_metric_tracker_log_average_value_with_prefix(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        tracker = ScalarMetricTracker()
        tracker.update({"metric0": 3, "metric1": 12})
        tracker.update({"metric0": 1, "metric1": 10})
        tracker.log_average_value(prefix="train/")
        assert caplog.messages[0] == "train/metric0: 2.000000"
        assert caplog.messages[1] == "train/metric1: 11.000000"


def test_scalar_metric_tracker_log_average_value_with_engine() -> None:
    tracker = ScalarMetricTracker()
    tracker.update({"metric0": 3, "metric1": 12})
    tracker.update({"metric0": 1, "metric1": 10})
    engine = Mock(spec=BaseEngine)
    engine.epoch = 0
    tracker.log_average_value(engine=engine)
    engine.log_metrics.assert_called_once_with({"metric0": 2, "metric1": 11}, step=EpochStep(0))


def test_scalar_metric_tracker_log_average_value_with_engine_and_prefix() -> None:
    tracker = ScalarMetricTracker()
    tracker.update({"metric0": 3, "metric1": 12})
    tracker.update({"metric0": 1, "metric1": 10})
    engine = Mock(spec=BaseEngine)
    engine.epoch = 0
    tracker.log_average_value(engine=engine, prefix="train/")
    engine.log_metrics.assert_called_once_with(
        {"train/metric0": 2, "train/metric1": 11}, step=EpochStep(0)
    )
