r"""This module defines a image folder dataset."""

__all__ = ["ImageFolderDataset"]

from unittest.mock import Mock

from gravitorch import constants as ct
from gravitorch.utils.integrations import (
    check_pillow,
    check_torchvision,
    is_torchvision_available,
)

if is_torchvision_available():
    from torchvision.datasets import ImageFolder
else:
    ImageFolder = Mock  # pragma: no cover


class ImageFolderDataset(ImageFolder):
    r"""Implements a dataset that returns a dict instead of a tuple.

    This dataset extends the ``torchvision.datasets.ImageFolder`` class.
    """

    def __init__(self, *args, **kwargs):
        check_torchvision()
        check_pillow()
        super().__init__(*args, **kwargs)

    def __getitem__(self, index: int) -> dict:
        r"""Get the image and the target of the index-th example.

        Args:
            index (int): Specifies the index of the example.

        Returns:
            dict: A dictionary with the image and the target of the
                ``index``-th example.
        """
        img, target = super().__getitem__(index)
        return {ct.INPUT: img, ct.TARGET: target}
