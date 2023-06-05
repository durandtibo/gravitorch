from unittest.mock import patch

from objectory import OBJECT_TARGET
from pytest import raises

from gravitorch.testing import torchvision_available
from gravitorch.transforms.vision import create_compose
from gravitorch.utils.imports import is_torchvision_available

if is_torchvision_available():
    from torchvision import transforms


##########################
#     create_compose     #
##########################


@torchvision_available
def test_create_compose_1() -> None:
    transform = create_compose([{OBJECT_TARGET: "torchvision.transforms.CenterCrop", "size": 10}])
    assert isinstance(transform, transforms.Compose)
    assert len(transform.transforms) == 1
    assert isinstance(transform.transforms[0], transforms.CenterCrop)


@torchvision_available
def test_create_compose_2() -> None:
    transform = create_compose(
        [
            {OBJECT_TARGET: "torchvision.transforms.CenterCrop", "size": 10},
            transforms.PILToTensor(),
        ]
    )
    assert isinstance(transform, transforms.Compose)
    assert len(transform.transforms) == 2
    assert isinstance(transform.transforms[0], transforms.CenterCrop)
    assert isinstance(transform.transforms[1], transforms.PILToTensor)


def test_create_compose_no_torchvision() -> None:
    with patch("gravitorch.utils.imports.is_torchvision_available", lambda *args: False):
        with raises(RuntimeError, match="`torchvision` package is required but not installed."):
            create_compose([])
