from typing import Union

from objectory import OBJECT_TARGET
from pytest import mark

from gravitorch import constants as ct
from gravitorch.models.metrics import BaseMetric, CategoricalAccuracy, setup_metric

##################################
#     Tests for setup_metric     #
##################################


@mark.parametrize(
    "metric",
    (
        CategoricalAccuracy(mode=ct.TRAIN),
        {OBJECT_TARGET: "gravitorch.models.metrics.CategoricalAccuracy", "mode": ct.TRAIN},
    ),
)
def test_setup_metric(metric: Union[BaseMetric, dict]):
    assert isinstance(setup_metric(metric), CategoricalAccuracy)
