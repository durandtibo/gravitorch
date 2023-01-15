__all__ = ["EpochRandomIterDataPipeCreator"]

import logging
from collections.abc import Sequence
from typing import Optional

from objectory import OBJECT_TARGET, factory
from torch.utils.data import IterDataPipe

from gravitorch import distributed as dist
from gravitorch.creators.datapipe.base import BaseIterDataPipeCreator
from gravitorch.engines.base import BaseEngine

logger = logging.getLogger(__name__)


class EpochRandomIterDataPipeCreator(BaseIterDataPipeCreator):
    r"""Implements an ``IterDataPipe`` creator to create an ``IterDataPipe``
    object where its random seed is controlled by an engine.

    Given an engine, the random seed is set based on the engine random
    seed, the current epoch value, the maximum number of epochs and
    the distributed rank.

    Args:
        config (dict): Specifies the configuration of the
            ``IterDataPipe``.
        random_seed_key (str, optional): Specifies the key in the
            configuration which is used to indicate the random seed
            of the ``IterDataPipe``. If this key exists in ``config``,
            it will be replaced by a new value, based on the engine
            state.
    """

    def __init__(self, config: dict, random_seed_key: str = "random_seed"):
        self._config = config
        self._random_seed_key = str(random_seed_key)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(config={self._config}, "
            f"random_seed_key={self._random_seed_key})"
        )

    def create(
        self, engine: Optional[BaseEngine] = None, source_inputs: Optional[Sequence] = None
    ) -> IterDataPipe:
        if engine is None:
            raise ValueError(
                "engine cannot be None because the epoch value is used to create the "
                "IterDataPipe object"
            )
        source_inputs = source_inputs or tuple()
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
