__all__ = ["MNISTDataset"]

from unittest.mock import Mock

from gravitorch import constants as ct
from gravitorch.utils.integrations import check_torchvision, is_torchvision_available

if is_torchvision_available():
    from torchvision.datasets import MNIST
else:
    MNIST = Mock  # pragma: no cover


class MNISTDataset(MNIST):
    r"""Updated MNIST dataset to return a dict instead of a tuple."""

    def __init__(self, *args, **kwargs):
        check_torchvision()
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
