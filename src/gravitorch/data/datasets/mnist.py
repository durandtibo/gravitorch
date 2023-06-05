__all__ = ["MNIST", "get_default_transform"]

from collections.abc import Callable
from pathlib import Path
from typing import Union
from unittest.mock import Mock

from gravitorch import constants as ct
from gravitorch.utils.factory import setup_object
from gravitorch.utils.imports import check_torchvision, is_torchvision_available
from gravitorch.utils.path import sanitize_path

if is_torchvision_available():
    from torchvision import transforms
    from torchvision.datasets import MNIST as MNIST_
    from torchvision.transforms import Compose
else:
    MNIST_, Compose = Mock, "Compose"  # pragma: no cover


class MNIST(MNIST_):
    r"""Updated MNIST dataset to return a dict instead of a tuple."""

    def __init__(
        self,
        root: Union[Path, str],
        train: bool = True,
        transform: Union[Callable, dict, None] = None,
        target_transform: Union[Callable, dict, None] = None,
        download: bool = False,
    ) -> None:
        check_torchvision()
        super().__init__(
            root=sanitize_path(root).as_posix(),
            train=train,
            transform=setup_object(transform),
            target_transform=setup_object(target_transform),
            download=download,
        )

    def __getitem__(self, index: int) -> dict:
        r"""Get the image and the target of the index-th example.

        Args:
        ----
            index (int): Specifies the index of the example.

        Returns:
        -------
            dict: A dictionary with the image and the target of the
                ``index``-th example.
        """
        img, target = super().__getitem__(index)
        return {ct.INPUT: img, ct.TARGET: target}


def get_default_transform() -> Compose:
    r"""Gets the default transform for MNIST dataset.

    Returns:
        ``torchvision.transforms.Compose``: The default transforms.

    Raises:
        RuntimeError if ``torchvision`` is not installed.
    """
    check_torchvision()
    return Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
