from unittest.mock import Mock

from objectory import OBJECT_TARGET
from torch import nn

from gravitorch.nn.init import BaseInitializer, NoOpInitializer, SequentialInitializer

###########################################
#     Tests for SequentialInitializer     #
###########################################


def test_sequential_initializer_str() -> None:
    assert str(SequentialInitializer([])).startswith("SequentialInitializer")


def test_sequential_initializer_init() -> None:
    initializer = SequentialInitializer(
        [
            NoOpInitializer(),
            {OBJECT_TARGET: "gravitorch.nn.init.NoOpInitializer"},
        ]
    )
    assert isinstance(initializer._initializers[0], NoOpInitializer)
    assert isinstance(initializer._initializers[1], NoOpInitializer)


def test_sequential_initializer_initialize() -> None:
    engine = Mock()
    engine.model = nn.Linear(4, 5)
    initializer1 = Mock(spec=BaseInitializer)
    initializer2 = Mock(spec=BaseInitializer)
    initializer = SequentialInitializer([initializer1, initializer2])
    initializer.initialize(engine)
    initializer1.initialize.assert_called_once_with(engine)
    initializer2.initialize.assert_called_once_with(engine)
