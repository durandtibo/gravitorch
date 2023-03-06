from objectory import OBJECT_TARGET

from gravitorch.utils.parameter_initializers import (
    NoParameterInitializer,
    setup_parameter_initializer,
)

#################################################
#     Tests for setup_parameter_initializer     #
#################################################


def test_setup_parameter_initializer_none() -> None:
    assert isinstance(setup_parameter_initializer(None), NoParameterInitializer)


def test_setup_parameter_initializer_dict() -> None:
    assert isinstance(
        setup_parameter_initializer(
            {OBJECT_TARGET: "gravitorch.utils.parameter_initializers.NoParameterInitializer"}
        ),
        NoParameterInitializer,
    )


def test_setup_parameter_initializer_object() -> None:
    initializer = NoParameterInitializer()
    assert setup_parameter_initializer(initializer) is initializer
