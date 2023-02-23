from unittest.mock import patch

from pytest import raises

from gravitorch.utils.integrations import (
    check_accelerate,
    check_fairscale,
    check_matplotlib,
    check_pillow,
    check_tensorboard,
    check_torchvision,
    is_accelerate_available,
    is_fairscale_available,
    is_matplotlib_available,
    is_pillow_available,
    is_psutil_available,
    is_tensorboard_available,
    is_torchvision_available,
)

######################
#     accelerate     #
######################


def test_check_accelerate_with_package():
    with patch("gravitorch.utils.integrations.is_accelerate_available", lambda *args: True):
        check_accelerate()


def test_check_accelerate_without_package():
    with patch("gravitorch.utils.integrations.is_accelerate_available", lambda *args: False):
        with raises(RuntimeError):
            check_accelerate()


def test_is_accelerate_available():
    assert isinstance(is_accelerate_available(), bool)


#####################
#     fairscale     #
#####################


def test_check_fairscale_with_package():
    with patch("gravitorch.utils.integrations.is_fairscale_available", lambda *args: True):
        check_fairscale()


def test_check_fairscale_without_package():
    with patch("gravitorch.utils.integrations.is_fairscale_available", lambda *args: False):
        with raises(RuntimeError):
            check_fairscale()


def test_is_fairscale_available():
    assert isinstance(is_fairscale_available(), bool)


######################
#     matplotlib     #
######################


def test_check_matplotlib_with_package():
    with patch("gravitorch.utils.integrations.is_matplotlib_available", lambda *args: True):
        check_matplotlib()


def test_check_matplotlib_without_package():
    with patch("gravitorch.utils.integrations.is_matplotlib_available", lambda *args: False):
        with raises(RuntimeError):
            check_matplotlib()


def test_is_matplotlib_available():
    assert isinstance(is_matplotlib_available(), bool)


##################
#     pillow     #
##################


def test_check_pillow_with_package():
    with patch("gravitorch.utils.integrations.is_pillow_available", lambda *args: True):
        check_pillow()


def test_check_pillow_without_package():
    with patch("gravitorch.utils.integrations.is_pillow_available", lambda *args: False):
        with raises(RuntimeError):
            check_pillow()


def test_is_pillow_available():
    assert isinstance(is_pillow_available(), bool)


##################
#     psutil     #
##################


def test_is_psutil_available():
    assert isinstance(is_psutil_available(), bool)


#######################
#     tensorboard     #
#######################


def test_check_tensorboard_with_package():
    with patch("gravitorch.utils.integrations.is_tensorboard_available", lambda *args: True):
        check_tensorboard()


def test_check_tensorboard_without_package():
    with patch("gravitorch.utils.integrations.is_tensorboard_available", lambda *args: False):
        with raises(RuntimeError):
            check_tensorboard()


def test_is_tensorboard_available():
    assert isinstance(is_tensorboard_available(), bool)


#######################
#     torchvision     #
#######################


def test_check_torchvision_with_package():
    with patch("gravitorch.utils.integrations.is_torchvision_available", lambda *args: True):
        check_torchvision()


def test_check_torchvision_without_package():
    with patch("gravitorch.utils.integrations.is_torchvision_available", lambda *args: False):
        with raises(RuntimeError):
            check_torchvision()


def test_is_torchvision_available():
    assert isinstance(is_torchvision_available(), bool)
