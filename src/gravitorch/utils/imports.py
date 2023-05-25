r"""This module implements some utility functions to check if some
packages are available."""

__all__ = [
    "check_accelerate",
    "check_fairscale",
    "check_matplotlib",
    "check_pillow",
    "check_safetensors",
    "check_tensorboard",
    "check_torchdata",
    "check_torchvision",
    "is_accelerate_available",
    "is_fairscale_available",
    "is_matplotlib_available",
    "is_pillow_available",
    "is_psutil_available",
    "is_safetensors_available",
    "is_tensorboard_available",
    "is_torchdata_available",
    "is_torchvision_available",
]

from importlib.util import find_spec

######################
#     accelerate     #
######################


def check_accelerate() -> None:
    r"""Checks if the ``accelerate`` package is installed.

    Raises
    ------
        RuntimeError if the ``accelerate`` package is not installed.
    """
    if not is_accelerate_available():
        raise RuntimeError(
            "`accelerate` package is required but not installed. "
            "You can install `accelerate` package with the command:\n\n"
            "pip install accelerate\n"
        )


def is_accelerate_available() -> bool:
    r"""Indicates if the ``accelerate`` package is installed or not.

    https://huggingface.co/docs/accelerate/index.html
    """
    return find_spec("accelerate") is not None


#####################
#     fairscale     #
#####################


def check_fairscale() -> None:
    r"""Checks if the ``fairscale`` package is installed.

    Raises
    ------
        RuntimeError if the ``fairscale`` package is not installed.
    """
    if not is_fairscale_available():
        raise RuntimeError(
            "`fairscale` package is required but not installed. "
            "You can install `fairscale` package with the command:\n\n"
            "pip install fairscale\n"
        )


def is_fairscale_available() -> bool:
    r"""Indicates if the ``fairscale`` package is installed or not."""
    return find_spec("fairscale") is not None


######################
#     matplotlib     #
######################


def check_matplotlib() -> None:
    r"""Checks if the ``matplotlib`` package is installed.

    Raises
    ------
        RuntimeError if the ``matplotlib`` package is not installed.
    """
    if not is_matplotlib_available():
        raise RuntimeError(
            "`matplotlib` package is required but not installed. "
            "You can install `matplotlib` package with the command:\n\n"
            "pip install matplotlib\n"
        )


def is_matplotlib_available() -> bool:
    r"""Indicates if the ``matplotlib`` package is installed or not."""
    return find_spec("matplotlib") is not None


##################
#     pillow     #
##################


def check_pillow() -> None:
    r"""Checks if the pillow package is installed.

    Raises
    ------
        RuntimeError if the ``pillow`` package is not installed.
    """
    if not is_pillow_available():
        raise RuntimeError(
            "`pillow` package is required but not installed. "
            "You can install `pillow` package with the command:\n\n"
            "pip install pillow\n"
        )


def is_pillow_available() -> bool:
    r"""Indicates if the ``pillow`` package is installed or not."""
    return find_spec("PIL") is not None


##################
#     psutil     #
##################


def is_psutil_available() -> bool:
    r"""Indicates if the ``psutil`` package is installed or not."""
    return find_spec("psutil") is not None


#######################
#     safetensors     #
#######################


def check_safetensors() -> None:
    r"""Checks if the ``safetensors`` package is installed.

    Raises
    ------
        RuntimeError if the ``safetensors`` package is not installed.
    """
    if not is_safetensors_available():
        raise RuntimeError(
            "`safetensors` package is required but not installed. "
            "You can install `safetensors` package with the command:\n\n"
            "pip install safetensors\n"
        )


def is_safetensors_available() -> bool:
    r"""Indicates if the ``safetensors`` package is installed or not.

    GitHub: https://github.com/huggingface/safetensors
    Docs: https://huggingface.co/docs/safetensors/index
    """
    return find_spec("safetensors") is not None


#######################
#     tensorboard     #
#######################


def check_tensorboard() -> None:
    r"""Checks if the ``tensorboard`` package is installed.

    Raises
    ------
        RuntimeError if the ``tensorboard`` package is not installed.
    """
    if not is_tensorboard_available():
        raise RuntimeError(
            "`tensorboard` package is required but not installed. "
            "You can install `tensorboard` package with the command:\n\n"
            "pip install tensorboard\n"
        )


def is_tensorboard_available() -> bool:
    r"""Indicates if the ``tensorboard`` package is installed or not."""
    return find_spec("tensorboard") is not None


#####################
#     torchdata     #
#####################


def check_torchdata() -> None:
    r"""Checks if the ``torchdata`` package is installed.

    Raises
    ------
        RuntimeError if the ``torchdata`` package is not installed.
    """
    if not is_torchdata_available():
        raise RuntimeError(
            "`torchdata` package is required but not installed. "
            "You can install `torchdata` package with the command:\n\n"
            "pip install torchdata\n"
        )


def is_torchdata_available() -> bool:
    r"""Indicates if the ``torchdata`` package is installed or not.

    https://pytorch.org/vision/stable/index.html
    """
    return find_spec("torchdata") is not None


#######################
#     torchvision     #
#######################


def check_torchvision() -> None:
    r"""Checks if the ``torchvision`` package is installed.

    Raises
    ------
        RuntimeError if the ``torchvision`` package is not installed.
    """
    if not is_torchvision_available():
        raise RuntimeError(
            "`torchvision` package is required but not installed. "
            "You can install `torchvision` package with the command:\n\n"
            "pip install torchvision\n"
        )


def is_torchvision_available() -> bool:
    r"""Indicates if the ``torchvision`` package is installed or not.

    https://pytorch.org/vision/stable/index.html
    """
    return find_spec("torchvision") is not None
