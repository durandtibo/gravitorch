r"""This module implements some utility functions to check if some
packages are available."""

from __future__ import annotations

__all__ = [
    "check_accelerate",
    "check_fairscale",
    "check_matplotlib",
    "check_pillow",
    "check_psutil",
    "check_startorch",
    "check_tensorboard",
    "check_torchdata",
    "check_torchvision",
    "check_tqdm",
    "is_accelerate_available",
    "is_fairscale_available",
    "is_matplotlib_available",
    "is_pillow_available",
    "is_psutil_available",
    "is_startorch_available",
    "is_tensorboard_available",
    "is_torchdata_available",
    "is_torchvision_available",
    "is_tqdm_available",
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

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.utils.imports import check_accelerate
        >>> check_accelerate()  # doctest: +SKIP
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

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.utils.imports import is_accelerate_available
        >>> is_accelerate_available()
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

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.utils.imports import check_fairscale
        >>> check_fairscale()  # xdoctest: +SKIP
    """
    if not is_fairscale_available():
        raise RuntimeError(
            "`fairscale` package is required but not installed. "
            "You can install `fairscale` package with the command:\n\n"
            "pip install fairscale\n"
        )


def is_fairscale_available() -> bool:
    r"""Indicates if the ``fairscale`` package is installed or not.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.utils.imports import is_fairscale_available
        >>> is_fairscale_available()
    """
    return find_spec("fairscale") is not None


######################
#     matplotlib     #
######################


def check_matplotlib() -> None:
    r"""Checks if the ``matplotlib`` package is installed.

    Raises
    ------
        RuntimeError if the ``matplotlib`` package is not installed.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.utils.imports import check_matplotlib
        >>> check_matplotlib()  # doctest: +SKIP
    """
    if not is_matplotlib_available():
        raise RuntimeError(
            "`matplotlib` package is required but not installed. "
            "You can install `matplotlib` package with the command:\n\n"
            "pip install matplotlib\n"
        )


def is_matplotlib_available() -> bool:
    r"""Indicates if the ``matplotlib`` package is installed or not.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.utils.imports import is_matplotlib_available
        >>> is_matplotlib_available()
    """
    return find_spec("matplotlib") is not None


##################
#     pillow     #
##################


def check_pillow() -> None:
    r"""Checks if the pillow package is installed.

    Raises
    ------
        RuntimeError if the ``pillow`` package is not installed.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.utils.imports import check_pillow
        >>> check_pillow()  # doctest: +SKIP
    """
    if not is_pillow_available():
        raise RuntimeError(
            "`pillow` package is required but not installed. "
            "You can install `pillow` package with the command:\n\n"
            "pip install pillow\n"
        )


def is_pillow_available() -> bool:
    r"""Indicates if the ``pillow`` package is installed or not.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.utils.imports import is_pillow_available
        >>> is_pillow_available()
    """
    return find_spec("PIL") is not None


##################
#     psutil     #
##################


def check_psutil() -> None:
    r"""Checks if the ``psutil`` package is installed.

    Raises
    ------
        RuntimeError if the ``psutil`` package is not installed.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.utils.imports import check_psutil
        >>> check_psutil()  # doctest: +SKIP
    """
    if not is_psutil_available():
        raise RuntimeError(
            "`psutil` package is required but not installed. "
            "You can install `psutil` package with the command:\n\n"
            "pip install psutil\n"
        )


def is_psutil_available() -> bool:
    r"""Indicates if the ``psutil`` package is installed or not.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.utils.imports import is_psutil_available
        >>> is_psutil_available()
    """
    return find_spec("psutil") is not None


#####################
#     startorch     #
#####################


def check_startorch() -> None:
    r"""Checks if the ``startorch`` package is installed.

    Raises
    ------
        RuntimeError if the ``startorch`` package is not installed.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.utils.imports import check_startorch
        >>> check_startorch()  # doctest: +SKIP
    """
    if not is_startorch_available():
        raise RuntimeError(
            "`startorch` package is required but not installed. "
            "You can install `startorch` package with the command:\n\n"
            "pip install startorch\n"
        )


def is_startorch_available() -> bool:
    r"""Indicates if the ``startorch`` package is installed or not.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.utils.imports import is_startorch_available
        >>> is_startorch_available()
    """
    return find_spec("startorch") is not None


#######################
#     tensorboard     #
#######################


def check_tensorboard() -> None:
    r"""Checks if the ``tensorboard`` package is installed.

    Raises
    ------
        RuntimeError if the ``tensorboard`` package is not installed.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.utils.imports import check_tensorboard
        >>> check_tensorboard()  # doctest: +SKIP
    """
    if not is_tensorboard_available():
        raise RuntimeError(
            "`tensorboard` package is required but not installed. "
            "You can install `tensorboard` package with the command:\n\n"
            "pip install tensorboard\n"
        )


def is_tensorboard_available() -> bool:
    r"""Indicates if the ``tensorboard`` package is installed or not.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.utils.imports import is_tensorboard_available
        >>> is_tensorboard_available()
    """
    return find_spec("tensorboard") is not None


#####################
#     torchdata     #
#####################


def check_torchdata() -> None:
    r"""Checks if the ``torchdata`` package is installed.

    Raises
    ------
        RuntimeError if the ``torchdata`` package is not installed.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.utils.imports import check_torchdata
        >>> check_torchdata()  # doctest: +SKIP
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

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.utils.imports import is_torchdata_available
        >>> is_torchdata_available()
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

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.utils.imports import check_torchvision
        >>> check_torchvision()  # doctest: +SKIP
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

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.utils.imports import is_torchvision_available
        >>> is_torchvision_available()
    """
    return find_spec("torchvision") is not None


#######################
#     tqdm     #
#######################


def check_tqdm() -> None:
    r"""Checks if the ``tqdm`` package is installed.

    Raises
    ------
        RuntimeError if the ``tqdm`` package is not installed.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.utils.imports import check_tqdm
        >>> check_tqdm()  # doctest: +SKIP
    """
    if not is_tqdm_available():
        raise RuntimeError(
            "`tqdm` package is required but not installed. "
            "You can install `tqdm` package with the command:\n\n"
            "pip install tqdm\n"
        )


def is_tqdm_available() -> bool:
    r"""Indicates if the ``tqdm`` package is installed or not.

    https://github.com/tqdm/tqdm

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.utils.imports import is_tqdm_available
        >>> is_tqdm_available()
    """
    return find_spec("tqdm") is not None
