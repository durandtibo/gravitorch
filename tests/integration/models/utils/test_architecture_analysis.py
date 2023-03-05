from torch import nn

from gravitorch.models import VanillaModel
from gravitorch.models.criteria import VanillaLoss
from gravitorch.models.utils import analyze_model_architecture
from gravitorch.models.utils.architecture_analysis import (
    analyze_model_network_architecture,
)
from gravitorch.testing import create_dummy_engine

################################################
#     Tests for analyze_model_architecture     #
################################################


def test_analyze_model_architecture() -> None:
    engine = create_dummy_engine()
    analyze_model_architecture(model=nn.Linear(4, 6), engine=engine)
    assert engine.get_history("model.num_parameters").get_last_value() == 30
    assert engine.get_history("model.num_learnable_parameters").get_last_value() == 30


########################################################
#     Tests for analyze_model_network_architecture     #
########################################################


def test_analyze_model_network_architecture() -> None:
    engine = create_dummy_engine()
    analyze_model_network_architecture(
        model=VanillaModel(
            network=nn.Linear(4, 6),
            criterion=VanillaLoss(criterion=nn.CrossEntropyLoss()),
        ),
        engine=engine,
    )
    assert engine.get_history("model.network.num_parameters").get_last_value() == 30
    assert engine.get_history("model.network.num_learnable_parameters").get_last_value() == 30


def test_analyze_model_network_architecture_no_network() -> None:
    engine = create_dummy_engine()
    analyze_model_network_architecture(
        model=nn.Linear(4, 6),
        engine=engine,
    )
    assert not engine.has_history("model.network.num_parameters")
    assert not engine.has_history("model.network.num_learnable_parameters")
