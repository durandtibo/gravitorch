from objectory import OBJECT_TARGET

from gravitorch.nn.init import NoOpInitializer, setup_initializer

#######################################
#     Tests for setup_initializer     #
#######################################


def test_setup_initializer_none() -> None:
    assert isinstance(setup_initializer(None), NoOpInitializer)


def test_setup_initializer_dict() -> None:
    assert isinstance(
        setup_initializer({OBJECT_TARGET: "gravitorch.nn.init.NoOpInitializer"}),
        NoOpInitializer,
    )


def test_setup_initializer_object() -> None:
    initializer = NoOpInitializer()
    assert setup_initializer(initializer) is initializer
