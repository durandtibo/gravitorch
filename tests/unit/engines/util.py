from typing import Optional, Union

import torch
from objectory import OBJECT_TARGET
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import Dataset, IterableDataset
from torch.utils.data.dataloader import default_collate

from gravitorch import constants as ct
from gravitorch.creators.core import VanillaCoreCreator
from gravitorch.creators.dataloader import VanillaDataLoaderCreator
from gravitorch.datasources import BaseDataSource, DatasetDataSource
from gravitorch.engines import AlphaEngine, BaseEngine
from gravitorch.models import BaseModel


class FakeMapDataset(Dataset):
    def __init__(self, feature_dim: int = 4):
        self._feature_dim = feature_dim

    def __getitem__(self, item) -> dict:
        return {ct.INPUT: torch.ones(self._feature_dim) + item, ct.TARGET: 1}

    def __len__(self) -> int:
        return 8

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(num_examples={len(self):,}, feature_dim={self._feature_dim:,})"


class EmptyFakeMapDataset(Dataset):
    def __getitem__(self, item) -> dict:
        return {}

    def __len__(self) -> int:
        return 0

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(num_examples={len(self):,})"


class FakeIterableDataset(IterableDataset):
    def __init__(self, feature_dim: int = 4):
        self._feature_dim = feature_dim
        self._iteration = 0

    def __iter__(self):
        return self

    def __next__(self):
        self._iteration += 1
        if self._iteration > 8:
            self._iteration = 0
            raise StopIteration

        return {ct.INPUT: torch.ones(self._feature_dim) + self._iteration, ct.TARGET: 1}

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"


class FakeIterableWithLengthDataset(IterableDataset):
    def __len__(self) -> int:
        return 8


class EmptyFakeIterableDataset(IterableDataset):
    def __iter__(self):
        return self

    def __next__(self):
        raise StopIteration

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"


class EmptyFakeDataSource(DatasetDataSource):
    r"""By default, the ``DatasetDataSource`` is empty."""


class FakeDataSource(DatasetDataSource):
    def __init__(
        self,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        batch_size: Optional[int] = 1,
    ):
        if train_dataset is None:
            train_dataset = FakeMapDataset()
        if eval_dataset is None:
            eval_dataset = FakeMapDataset()
        super().__init__(
            datasets={"train": train_dataset, "eval": eval_dataset},
            data_loader_creators={
                "train": VanillaDataLoaderCreator(
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=default_collate,
                ),
                "eval": VanillaDataLoaderCreator(
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=default_collate,
                ),
            },
        )


class FakeModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, batch: dict):
        return {
            ct.LOSS: self.criterion(input=self.linear(batch[ct.INPUT]), target=batch[ct.TARGET])
        }


class FakeModelWithNaN(BaseModel):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Linear(4, 6)

    def forward(self, batch: dict):
        return {ct.LOSS: torch.tensor(float("nan"))}


def create_engine(
    data_source: Union[BaseDataSource, dict, None] = None,
    model: Union[nn.Module, dict, None] = None,
    optimizer: Union[Optimizer, dict, None] = None,
    device: Optional[torch.device] = None,
    **kwargs,
) -> BaseEngine:
    data_source = data_source or FakeDataSource(batch_size=2)
    model = model or FakeModel()
    optimizer = optimizer or {OBJECT_TARGET: "torch.optim.sgd.SGD", "lr": 0.01}
    return AlphaEngine(
        core_creator=VanillaCoreCreator(
            data_source=data_source,
            model=model.to(device=device),
            optimizer=optimizer,
        ),
        **kwargs,
    )
