import torch
from torch import nn

from gravitorch.nn.init import NoOpInitializer

#####################################
#     Tests for NoOpInitializer     #
#####################################


def test_noop_initializer_str() -> None:
    assert str(NoOpInitializer()).startswith("NoOpInitializer(")


def test_noop_initializer() -> None:
    module = nn.Linear(4, 6)
    nn.init.ones_(module.weight.data)
    NoOpInitializer().initialize(module)
    assert module.weight.data.equal(torch.ones(6, 4))
