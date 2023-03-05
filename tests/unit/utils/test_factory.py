from objectory import OBJECT_TARGET
from torch.distributions import Uniform

from gravitorch.utils.factory import setup_distribution

########################################
#     Tests for setup_distribution     #
########################################


def test_setup_distribution_object() -> None:
    distribution = Uniform(0.0, 2.0)
    assert setup_distribution(distribution) is distribution


def test_setup_distribution_dict() -> None:
    assert isinstance(
        setup_distribution({OBJECT_TARGET: "torch.distributions.Uniform", "low": 0.0, "high": 2.0}),
        Uniform,
    )
