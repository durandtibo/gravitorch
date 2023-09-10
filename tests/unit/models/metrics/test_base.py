import logging
from collections import defaultdict

from objectory import OBJECT_TARGET
from pytest import LogCaptureFixture

from gravitorch import constants as ct
from gravitorch.models.metrics import CategoricalAccuracy, setup_metric

##################################
#     Tests for setup_metric     #
##################################


def test_setup_metric_object() -> None:
    metric = CategoricalAccuracy(mode=ct.TRAIN)
    assert setup_metric(metric) is metric


def test_setup_metric_config() -> None:
    assert isinstance(
        setup_metric(
            {OBJECT_TARGET: "gravitorch.models.metrics.CategoricalAccuracy", "mode": ct.TRAIN}
        ),
        CategoricalAccuracy,
    )


def test_setup_metric_incorrect_type(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(level=logging.WARNING):
        assert isinstance(setup_metric({OBJECT_TARGET: "collections.defaultdict"}), defaultdict)
        assert caplog.messages
