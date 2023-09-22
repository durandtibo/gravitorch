from __future__ import annotations

__all__ = ["BaseDatasetCreator", "setup_dataset_creator"]

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

from objectory import AbstractFactory
from torch.utils.data import Dataset

from gravitorch.utils.format import str_target_object

if TYPE_CHECKING:
    from gravitorch.engines import BaseEngine

logger = logging.getLogger(__name__)

T = TypeVar("T")


class BaseDatasetCreator(Generic[T], ABC, metaclass=AbstractFactory):
    r"""Define the base class to implement a dataset creator.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.creators.dataset import DatasetCreator
        >>> creator = DatasetCreator(
        ...     {
        ...         "_target_": "gravitorch.datasets.DummyMultiClassDataset",
        ...         "num_examples": 10,
        ...         "num_classes": 2,
        ...         "feature_size": 4,
        ...     }
        ... )
        >>> creator
        DatasetCreator(
          cache=False
          dataset={'_target_': 'gravitorch.datasets.DummyMultiClassDataset', 'num_examples': 10, 'num_classes': 2, 'feature_size': 4}
        )
        >>> creator.create()
        DummyMultiClassDataset(num_examples=10, num_classes=2, feature_size=4, noise_std=0.2, ...)
    """

    @abstractmethod
    def create(self, engine: BaseEngine | None = None) -> Dataset[T]:
        r"""Create a dataset.

        Args:
        ----
            engine (``gravitorch.engines.BaseEngine`` or ``None``,
                optional): Specifies an engine. Default: ``None``

        Returns:
        -------
            ``torch.utils.data.Dataset``: The created dataset.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.creators.dataset import DatasetCreator
            >>> creator = DatasetCreator(
            ...     {
            ...         "_target_": "gravitorch.datasets.DummyMultiClassDataset",
            ...         "num_examples": 10,
            ...         "num_classes": 2,
            ...         "feature_size": 4,
            ...     }
            ... )
            >>> creator.create()
            DummyMultiClassDataset(num_examples=10, num_classes=2, feature_size=4, noise_std=0.2, ...)
        """


def setup_dataset_creator(creator: BaseDatasetCreator | dict) -> BaseDatasetCreator:
    r"""Sets up the dataset creator.

    The dataset creator is instantiated from its configuration by
    using the ``BaseDatasetCreator`` factory function.

    Args:
    ----
        creator (``BaseDatasetCreator`` or dict): Specifies the
            dataset creator or its configuration.

    Returns:
    -------
        ``BaseDatasetCreator``: The instantiated dataset creator.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.creators.dataset import setup_dataset_creator
        >>> creator = setup_dataset_creator(
        ...     {
        ...         "_target_": "gravitorch.creators.dataset.DatasetCreator",
        ...         "dataset": {
        ...             "_target_": "gravitorch.datasets.DummyMultiClassDataset",
        ...             "num_examples": 10,
        ...             "num_classes": 2,
        ...             "feature_size": 4,
        ...         },
        ...     }
        ... )
        >>> creator
        DatasetCreator(
          cache=False
          dataset={'_target_': 'gravitorch.datasets.DummyMultiClassDataset', 'num_examples': 10, 'num_classes': 2, 'feature_size': 4}
        )
    """
    if isinstance(creator, dict):
        logger.info(
            "Initializing the dataset creator from its configuration... "
            f"{str_target_object(creator)}"
        )
        creator = BaseDatasetCreator.factory(**creator)
    return creator
