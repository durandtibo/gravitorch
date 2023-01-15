r"""This module defines some functionalities to instantiate dynamically a
``torch.utils.data.Dataset`` object from its configuration."""

__all__ = ["setup_dataset"]

import logging
from typing import Optional, TypeVar, Union

from objectory import factory
from torch.utils.data import Dataset

from gravitorch.utils.format import str_target_object

logger = logging.getLogger(__name__)

T = TypeVar("T")


def setup_dataset(dataset: Union[Dataset, dict, None]) -> Optional[Dataset]:
    r"""Sets up a dataset.

    Args:
        dataset (``Dataset`` or dict or ``None``): Specifies the
            dataset or its configuration (dictionary). If a
            configuration is given, a dataset object is instantiated
            from the configuration.

    Returns:
        ``torch.utils.data.Dataset`` or ``None``: A dataset object or
            ``None`` if there is no dataset.

    Example usage:

    .. code-block:: python

        >>> from gravitorch.data.datasets import setup_dataset
        >>> mnist = setup_dataset(
        ...     {'_target_': 'torchvision.datasets.MNIST', 'root': '/my/path/', 'download': True},
        ... )
        >>> mnist
        Dataset MNIST
            Number of datapoints: 60000
            Root location: tmp
            Split: Train
        >>> setup_dataset(mnist)  # Do nothing because the dataset is already instantiated
        Dataset MNIST
            Number of datapoints: 60000
            Root location: tmp
            Split: Train
        >>> setup_dataset(None)
    """
    if isinstance(dataset, dict):
        logger.info(
            f"Initializing a dataset from its configuration... {str_target_object(dataset)}"
        )
        dataset = factory(**dataset)
    return dataset
