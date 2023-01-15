from unittest.mock import Mock

from objectory import OBJECT_TARGET
from torch import nn

from gravitorch.utils.parameter_initializers import (
    NoParameterInitializer,
    SequentialParameterInitializer,
)

####################################################
#     Tests for SequentialParameterInitializer     #
####################################################


def test_sequential_parameter_initializer_str():
    assert str(SequentialParameterInitializer([])).startswith("SequentialParameterInitializer")


def test_sequential_parameter_initializer_init():
    parameter_initializer = SequentialParameterInitializer(
        [
            NoParameterInitializer(),
            {OBJECT_TARGET: "gravitorch.utils.parameter_initializers.NoParameterInitializer"},
        ]
    )
    assert isinstance(parameter_initializer._parameter_initializers[0], NoParameterInitializer)
    assert isinstance(parameter_initializer._parameter_initializers[1], NoParameterInitializer)


def test_sequential_parameter_initializer_initialize():
    engine = Mock()
    engine.model = nn.Linear(4, 5)
    parameter_initializer1 = Mock()
    parameter_initializer2 = Mock()
    parameter_initializer = SequentialParameterInitializer(
        [parameter_initializer1, parameter_initializer2]
    )
    parameter_initializer.initialize(engine)
    parameter_initializer1.initialize.assert_called_once_with(engine)
    parameter_initializer2.initialize.assert_called_once_with(engine)
