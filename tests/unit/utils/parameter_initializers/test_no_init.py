from unittest.mock import Mock

import torch
from torch import nn

from gravitorch.utils.parameter_initializers import NoParameterInitializer

############################################
#     Tests for NoParameterInitializer     #
############################################


def test_no_model_parameter_initializer_str():
    assert str(NoParameterInitializer()).startswith("NoParameterInitializer(")


def test_no_model_parameter_initializer():
    initializer = NoParameterInitializer()
    engine = Mock()
    engine.model = nn.Linear(4, 6)
    nn.init.ones_(engine.model.weight.data)
    initializer.initialize(engine)
    assert engine.model.weight.data.equal(torch.ones(6, 4))
