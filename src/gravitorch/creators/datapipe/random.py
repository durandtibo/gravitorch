from __future__ import annotations

__all__ = ["EpochRandomDataPipeCreator"]

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, TypeVar

from coola.utils import str_indent, str_mapping
from objectory import OBJECT_TARGET, factory
from torch.utils.data import IterDataPipe, MapDataPipe

from gravitorch.creators.datapipe.base import BaseDataPipeCreator
from gravitorch.distributed import comm as dist
from gravitorch.utils.format import str_pretty_json

if TYPE_CHECKING:
    from gravitorch.engines import BaseEngine

logger = logging.getLogger(__name__)

T = TypeVar("T")


class EpochRandomDataPipeCreator(BaseDataPipeCreator[T]):
    r"""Implements a ``DataPipe`` creator to create a ``DataPipe`` object
    where its random seed is controlled by an engine.

    Given an engine, the random seed is set based on the engine random
    seed, the current epoch value, the maximum number of epochs and
    the distributed rank.

    Args:
    ----
        config (dict): Specifies the configuration of the
            ``DataPipe``.
        random_seed_key (str, optional): Specifies the key in the
            configuration which is used to indicate the random seed
            of the ``DataPipe``. If this key exists in ``config``,
            it will be replaced by a new value, based on the engine
            state.

    Example usage:

    .. code-block:: pycon

        >>> from torch.utils.data.datapipes.iter import IterableWrapper
        >>> from gravitorch.creators.datapipe import EpochRandomDataPipeCreator
        >>> from gravitorch.testing import create_dummy_engine
        >>> creator = EpochRandomDataPipeCreator(
        ...     {
        ...         "_target_": "gravitorch.datapipes.iter.TensorDictShuffler",
        ...         "datapipe": {
        ...             "_target_": "torch.utils.data.datapipes.iter.IterableWrapper",
        ...             "iterable": [1, 2, 3, 4, 5],
        ...         },
        ...         "random_seed": 42,
        ...     }
        ... )
        >>> creator
        EpochRandomDataPipeCreator(
          (config): {
              "_target_": "gravitorch.datapipes.iter.TensorDictShuffler",
              "datapipe": {
                "_target_": "torch.utils.data.datapipes.iter.IterableWrapper",
                "iterable": [
                  1,
                  2,
                  3,
                  4,
                  5
                ]
              },
              "random_seed": 42
            }
          (random_seed_key): random_seed
        )
        >>> engine = create_dummy_engine()
        >>> dp = creator.create(engine)
        >>> dp
        TensorDictShufflerIterDataPipe(
          (dim): 0
          (random_seed): 41
          (datapipe): {'_target_': 'torch.utils.data.datapipes.iter.IterableWrapper', 'iterable': [1, 2, 3, 4, 5]}
        )
    """

    def __init__(self, config: dict, random_seed_key: str = "random_seed") -> None:
        self._config = config
        self._random_seed_key = str(random_seed_key)

    def __repr__(self) -> str:
        args = str_indent(
            str_mapping(
                {
                    "config": str_pretty_json(self._config),
                    "random_seed_key": self._random_seed_key,
                }
            )
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def create(
        self, engine: BaseEngine | None = None, source_inputs: Sequence | None = None
    ) -> IterDataPipe[T] | MapDataPipe[T]:
        if engine is None:
            raise RuntimeError(
                "engine cannot be None because the epoch value is used to create the "
                "DataPipe object"
            )
        source_inputs = source_inputs or ()
        config = self._config.copy()  # Make a copy because the dict is modified below.
        target = config.pop(OBJECT_TARGET)
        config[self._random_seed_key] = (
            config.get(self._random_seed_key, engine.random_seed)
            + engine.epoch
            + engine.max_epochs * dist.get_rank()
        )
        logger.info(
            f"Set the random seed to {config[self._random_seed_key]}  [{OBJECT_TARGET}: {target}]"
        )
        return factory(target, *source_inputs, **config)
