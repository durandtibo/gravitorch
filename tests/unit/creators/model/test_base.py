from objectory import OBJECT_TARGET

from gravitorch.creators.model import VanillaModelCreator, setup_model_creator

#########################################
#     Tests for setup_model_creator     #
#########################################


def test_setup_model_creator_object():
    model_creator = VanillaModelCreator(model_config={})
    assert setup_model_creator(model_creator) is model_creator


def test_setup_model_creator_dict():
    assert isinstance(
        setup_model_creator(
            {
                OBJECT_TARGET: "gravitorch.creators.model.VanillaModelCreator",
                "model_config": {},
            }
        ),
        VanillaModelCreator,
    )
