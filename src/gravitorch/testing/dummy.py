r"""The code is in this module is designed to be used for testing purpose
only."""

__all__ = [
    "DummyDataset",
    "DummyIterableDataset",
    "DummyClassificationModel",
    "DummyDataSource",
    "create_dummy_engine",
]

from typing import Optional, Union

import torch
from objectory import OBJECT_TARGET
from torch.nn import CrossEntropyLoss, Linear, Module
from torch.optim import Optimizer
from torch.utils.data import Dataset, IterableDataset

from gravitorch import constants as ct
from gravitorch.datasources import BaseDataSource, DatasetDataSource
from gravitorch.engines import BaseEngine
from gravitorch.models import BaseModel


class DummyDataset(Dataset):
    r"""Implements a dummy map-style dataset for testing purpose.

    Args:
        feature_size (dim, optional): Specifies the feature size.
            Default: ``4``
        num_examples (dim, optional): Specifies the number of
            examples. Default: ``8``
    """

    def __init__(self, feature_size: int = 4, num_examples: int = 8):
        self._feature_size = int(feature_size)
        self._num_examples = int(num_examples)

    def __getitem__(self, item: int) -> dict:
        return {ct.INPUT: torch.ones(self._feature_size) + item, ct.TARGET: 1}

    def __len__(self) -> int:
        return self._num_examples

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(num_examples={self._feature_size:,}, "
            f"feature_size={self._feature_size:,})"
        )


class DummyIterableDataset(IterableDataset):
    r"""Implements a dummy iterable-style dataset for testing purpose.

    Args:
        feature_size (dim, optional): Specifies the feature size.
            Default: ``4``
        num_examples (dim, optional): Specifies the number of
            examples. Default: ``8``
        has_length (bool, optional): If ``True``, the length of the
            dataset is defined, otherwise it returns ``TypeError``.
            Default: ``False``
    """

    def __init__(self, feature_size: int = 4, num_examples: int = 8, has_length: bool = False):
        self._feature_size = int(feature_size)
        self._num_examples = int(num_examples)
        self._has_length = bool(has_length)
        self._iteration = 0

    def __iter__(self):
        self._iteration = 0
        return self

    def __next__(self):
        self._iteration += 1
        if self._iteration > self._num_examples:
            raise StopIteration

        return {ct.INPUT: torch.ones(self._feature_size) + self._iteration, ct.TARGET: 1}

    def __len__(self) -> int:
        if self._has_length:
            return self._num_examples
        raise TypeError

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(num_examples={self._feature_size:,}, "
            f"feature_size={self._feature_size:,})"
        )


class DummyDataSource(DatasetDataSource):
    r"""Implements a dummy data source for testing purpose.

    Args:
        train_dataset (``Dataset`` or ``None``, optional): Specifies
            the training dataset. If ``None``, a dummy map-style
            dataset is automatically created. Default: ``None``
        eval_dataset (``Dataset`` or ``None``, optional): Specifies
            the evaluation dataset. If ``None``, a dummy map-style
            dataset is automatically created. Default: ``None``
        batch_size (int, optional): Specifies the batch size.
            Default: ``1``
    """

    def __init__(
        self,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        batch_size: Optional[int] = 1,
    ):
        if train_dataset is None:
            train_dataset = DummyDataset()
        if eval_dataset is None:
            eval_dataset = DummyDataset()

        from gravitorch.creators.dataloader import VanillaDataLoaderCreator

        super().__init__(
            datasets={ct.TRAIN: train_dataset, ct.EVAL: eval_dataset},
            data_loader_creators={
                ct.TRAIN: VanillaDataLoaderCreator(batch_size=batch_size, shuffle=False),
                ct.EVAL: VanillaDataLoaderCreator(batch_size=batch_size, shuffle=False),
            },
        )


class DummyClassificationModel(BaseModel):
    r"""Implements a dummy classification model for testing purpose.

    Args:
        feature_size (dim, optional): Specifies the feature size.
            Default: ``4``
        num_classes (dim, optional): Specifies the number of classes.
            Default: ``3``
        loss_nan (bool, optional): If ``True``, the forward function
            returns a loss filled with a NaN value.
    """

    def __init__(self, feature_size: int = 4, num_classes: int = 3, loss_nan: bool = False):
        super().__init__()
        self.linear = Linear(feature_size, num_classes)
        self.criterion = CrossEntropyLoss()
        self._return_loss_nan = bool(loss_nan)

    def forward(self, batch: dict) -> dict:
        if self._return_loss_nan:
            return {ct.LOSS: torch.tensor(float("nan"))}
        return {ct.LOSS: self.criterion(self.linear(batch[ct.INPUT]), batch[ct.TARGET])}


def create_dummy_engine(
    data_source: Union[BaseDataSource, dict, None] = None,
    model: Union[Module, dict, None] = None,
    optimizer: Union[Optimizer, dict, None] = None,
    device: Optional[torch.device] = None,
    **kwargs,
) -> BaseEngine:
    r"""Creates an engine with dummy components for testing purpose.

    Args:
        data_source (``BaseDataSource`` or dict or ``None``): Specifies
            the data source or its configuration. If ``None``, a dummy
            data source is automatically created. Default: ``None``
        model (``Module`` or dict or ``None``): Specifies the model or
            its configuration. If ``None``, a dummy classification model
            is automatically created. Default: ``None``
        model (``Optimizer`` or dict or ``None``): Specifies the
            optimizer or its configuration. If ``None``, a SGD
            optimizer is automatically created. Default: ``None``
        device (``torch.device`` or ``None``): Specifies the target
            device. Default: ``None``
        **kwargs: Arbitrary keyword arguments.
    """
    data_source = data_source or DummyDataSource(batch_size=2)
    model = model or DummyClassificationModel()
    optimizer = optimizer or {OBJECT_TARGET: "torch.optim.SGD", "lr": 0.01}

    from gravitorch.creators.core import VanillaCoreCreator
    from gravitorch.engines import AlphaEngine

    return AlphaEngine(
        core_creator=VanillaCoreCreator(
            data_source=data_source,
            model=model.to(device=device),
            optimizer=optimizer,
        ),
        **kwargs,
    )
