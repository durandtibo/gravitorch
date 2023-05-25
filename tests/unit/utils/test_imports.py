from unittest.mock import patch

from pytest import raises

from gravitorch.utils.imports import (
    check_accelerate,
    check_fairscale,
    check_matplotlib,
    check_pillow,
    check_safetensors,
    check_tensorboard,
    check_torchdata,
    check_torchvision,
    is_accelerate_available,
    is_fairscale_available,
    is_matplotlib_available,
    is_pillow_available,
    is_psutil_available,
    is_safetensors_available,
    is_tensorboard_available,
    is_torchdata_available,
    is_torchvision_available,
)

######################
#     accelerate     #
######################


def test_check_accelerate_with_package() -> None:
    with patch("gravitorch.utils.imports.is_accelerate_available", lambda *args: True):
        check_accelerate()


def test_check_accelerate_without_package() -> None:
    with patch("gravitorch.utils.imports.is_accelerate_available", lambda *args: False):
        with raises(RuntimeError, match="`accelerate` package is required but not installed."):
            check_accelerate()


def test_is_accelerate_available() -> None:
    assert isinstance(is_accelerate_available(), bool)


#####################
#     fairscale     #
#####################


def test_check_fairscale_with_package() -> None:
    with patch("gravitorch.utils.imports.is_fairscale_available", lambda *args: True):
        check_fairscale()


def test_check_fairscale_without_package() -> None:
    with patch("gravitorch.utils.imports.is_fairscale_available", lambda *args: False):
        with raises(RuntimeError, match="`fairscale` package is required but not installed."):
            check_fairscale()


def test_is_fairscale_available() -> None:
    assert isinstance(is_fairscale_available(), bool)


######################
#     matplotlib     #
######################


def test_check_matplotlib_with_package() -> None:
    with patch("gravitorch.utils.imports.is_matplotlib_available", lambda *args: True):
        check_matplotlib()


def test_check_matplotlib_without_package() -> None:
    with patch("gravitorch.utils.imports.is_matplotlib_available", lambda *args: False):
        with raises(RuntimeError, match="`matplotlib` package is required but not installed."):
            check_matplotlib()


def test_is_matplotlib_available() -> None:
    assert isinstance(is_matplotlib_available(), bool)


##################
#     pillow     #
##################


def test_check_pillow_with_package() -> None:
    with patch("gravitorch.utils.imports.is_pillow_available", lambda *args: True):
        check_pillow()


def test_check_pillow_without_package() -> None:
    with patch("gravitorch.utils.imports.is_pillow_available", lambda *args: False):
        with raises(RuntimeError, match="`pillow` package is required but not installed."):
            check_pillow()


def test_is_pillow_available() -> None:
    assert isinstance(is_pillow_available(), bool)


##################
#     psutil     #
##################


def test_is_psutil_available() -> None:
    assert isinstance(is_psutil_available(), bool)


#######################
#     safetensors     #
#######################


def test_check_safetensors_with_package() -> None:
    with patch("gravitorch.utils.imports.is_safetensors_available", lambda *args: True):
        check_safetensors()


def test_check_safetensors_without_package() -> None:
    with patch("gravitorch.utils.imports.is_safetensors_available", lambda *args: False):
        with raises(RuntimeError, match="`safetensors` package is required but not installed."):
            check_safetensors()


def test_is_safetensors_available() -> None:
    assert isinstance(is_safetensors_available(), bool)


#######################
#     tensorboard     #
#######################


def test_check_tensorboard_with_package() -> None:
    with patch("gravitorch.utils.imports.is_tensorboard_available", lambda *args: True):
        check_tensorboard()


def test_check_tensorboard_without_package() -> None:
    with patch("gravitorch.utils.imports.is_tensorboard_available", lambda *args: False):
        with raises(RuntimeError, match="`tensorboard` package is required but not installed."):
            check_tensorboard()


def test_is_tensorboard_available() -> None:
    assert isinstance(is_tensorboard_available(), bool)


#######################
#     torchdata     #
#######################


def test_check_torchdata_with_package() -> None:
    with patch("gravitorch.utils.imports.is_torchdata_available", lambda *args: True):
        check_torchdata()


def test_check_torchdata_without_package() -> None:
    with patch("gravitorch.utils.imports.is_torchdata_available", lambda *args: False):
        with raises(RuntimeError, match="`torchdata` package is required but not installed."):
            check_torchdata()


def test_is_torchdata_available() -> None:
    assert isinstance(is_torchdata_available(), bool)


#######################
#     torchvision     #
#######################


def test_check_torchvision_with_package() -> None:
    with patch("gravitorch.utils.imports.is_torchvision_available", lambda *args: True):
        check_torchvision()


def test_check_torchvision_without_package() -> None:
    with patch("gravitorch.utils.imports.is_torchvision_available", lambda *args: False):
        with raises(RuntimeError, match="`torchvision` package is required but not installed."):
            check_torchvision()


def test_is_torchvision_available() -> None:
    assert isinstance(is_torchvision_available(), bool)
