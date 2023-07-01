from unittest.mock import Mock

from gravitorch.creators.optimizer import NoOptimizerCreator
from gravitorch.engines import BaseEngine

########################################
#     Tests for NoOptimizerCreator     #
########################################


def test_vanilla_optimizer_creator_str() -> None:
    assert str(NoOptimizerCreator()).startswith("NoOptimizerCreator(")


def test_vanilla_optimizer_creator_create_optimizer_config_none() -> None:
    creator = NoOptimizerCreator()
    assert creator.create(engine=Mock(spec=BaseEngine), model=Mock()) is None
