from objectory import OBJECT_TARGET

from gravitorch.creators.model import (
    ModelCreator,
    is_model_creator_config,
    setup_model_creator,
)

#############################################
#     Tests for is_model_creator_config     #
#############################################


def test_is_model_creator_config_true() -> None:
    assert is_model_creator_config(
        {
            OBJECT_TARGET: "gravitorch.creators.model.ModelCreator",
            "model_config": {"_target_": "gravitorch.testing.DummyClassificationModel"},
        }
    )


def test_is_model_creator_config_false() -> None:
    assert not is_model_creator_config({OBJECT_TARGET: "torch.nn.Identity"})


#########################################
#     Tests for setup_model_creator     #
#########################################


def test_setup_model_creator_object() -> None:
    model_creator = ModelCreator(model_config={})
    assert setup_model_creator(model_creator) is model_creator


def test_setup_model_creator_dict() -> None:
    assert isinstance(
        setup_model_creator(
            {
                OBJECT_TARGET: "gravitorch.creators.model.ModelCreator",
                "model_config": {},
            }
        ),
        ModelCreator,
    )
