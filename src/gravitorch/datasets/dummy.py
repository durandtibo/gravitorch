r"""This module defines some dummy datasets."""

from __future__ import annotations

__all__ = ["DummyMultiClassDataset"]

import logging

from torch.utils.data import Dataset

from gravitorch import constants as ct
from gravitorch.datasets.utils import log_box_dataset_class

logger = logging.getLogger(__name__)


class DummyMultiClassDataset(Dataset):
    r"""Implements a toy multi-class classification dataset in a map-
    style format.

    Args:
    ----
        num_examples (int, optional): Specifies the number of
            examples. Default: ``1000``
        num_classes (int, optional): Specifies the number of classes.
            Default: 50
        feature_size (int, optional): Specifies the feature size. The
            feature size has to be greater than the number of classes.
            Default: ``64``
        noise_std (float, optional): Specifies the standard deviation
            of the Gaussian noise. Default: ``0.2``
        random_seed (int, optional): Specifies the random seed used to
            initialize a ``torch.Generator`` object.
            Default: ``10169389905513828140``

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.datasets import DummyMultiClassDataset
        >>> dataset = DummyMultiClassDataset(num_examples=10, num_classes=5, feature_size=6)
        >>> dataset
        DummyMultiClassDataset(num_examples=10, num_classes=5, feature_size=6, noise_std=0.2, random_seed=10169389905513828140)
        >>> dataset[0]
        {'input': tensor([...]), 'target': tensor(...), 'name': '0'}
    """

    def __init__(
        self,
        num_examples: int = 1000,
        num_classes: int = 50,
        feature_size: int = 64,
        noise_std: float = 0.2,
        random_seed: int = 10169389905513828140,
    ) -> None:
        log_box_dataset_class(self)

        logger.info("Initializing the data creator...")
        # Import is here to avoid cyclic import
        from gravitorch.data.datacreators import HypercubeVertexDataCreator

        self._data_creator = HypercubeVertexDataCreator(
            num_examples=num_examples,
            num_classes=num_classes,
            feature_size=feature_size,
            noise_std=noise_std,
            random_seed=random_seed,
        )
        data = self._data_creator.create()

        self._targets = data[ct.TARGET]
        self._features = data[ct.INPUT]

    def __getitem__(self, item: int) -> dict:
        return {
            ct.INPUT: self._features[item],
            ct.TARGET: self._targets[item],
            ct.NAME: f"{item}",
        }

    def __len__(self) -> int:
        return self.num_examples

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"num_examples={self.num_examples:,}, "
            f"num_classes={self.num_classes:,}, "
            f"feature_size={self.feature_size:,}, "
            f"noise_std={self.noise_std:,}, "
            f"random_seed={self.random_seed})"
        )

    @property
    def num_examples(self) -> int:
        r"""``int``: The number of examples when the data are
        created."""
        return self._data_creator.num_examples

    @property
    def num_classes(self) -> int:
        r"""``int``: The number of classes when the data are created."""
        return self._data_creator.num_classes

    @property
    def feature_size(self) -> int:
        r"""``int``: The feature size when the data are created."""
        return self._data_creator.feature_size

    @property
    def noise_std(self) -> float:
        r"""``float``: The standard deviation of the Gaussian noise."""
        return self._data_creator.noise_std

    @property
    def random_seed(self) -> int:
        r"""int: The random seed used to initialize a
        ``torch.Generator`` object.
        """
        return self._data_creator.random_seed
