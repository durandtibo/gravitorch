__all__ = ["ZeroRedundancyOptimizerCreator"]

import logging
from typing import Optional

from objectory import OBJECT_TARGET
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn import Module
from tornado.util import import_object

from gravitorch import constants as ct
from gravitorch.creators.optimizer.base import BaseOptimizerCreator
from gravitorch.engines.base import BaseEngine
from gravitorch.handlers import ConsolidateOptimizerState
from gravitorch.utils.format import str_indent, to_pretty_json_str

logger = logging.getLogger(__name__)


class ZeroRedundancyOptimizerCreator(BaseOptimizerCreator):
    r"""Implements a zero redundancy optimizer (ZeRO) creator.

    Documentation of ``torch.distributed.optim.ZeroRedundancyOptimizer``:
    https://pytorch.org/docs/stable/distributed.optim.html#torch.distributed.optim.ZeroRedundancyOptimizer

    This creator wraps an arbitrary ``torch.optim.Optimizer``
    optimizer and shards its state as described by ZeRO
    (https://arxiv.org/abs/1910.02054).
    Note ZeRO works only for distributed training.

    Args:
        optimizer_config (dict): Specifies the optimizer
            configuration. The dictionary must have a key
            ``'_target_`` that indicates the optimizer to shard. The
            other keys are the arguments of the optimizer to shard.
        zero_kwargs (dict): Specifies some keyword arguments used to
            instantiate the
            ``torch.distributed.optim.ZeroRedundancyOptimizer``.
            Please read the documentation of
            ``torch.distributed.optim.ZeroRedundancyOptimizer`` to see
            the possible options. Note that it is not possible to set
            ``params`` and ``optimizer_class`` with a keyword argument.
        add_module_to_engine (bool, optional): If ``True``, the
            optimizer is added to the engine state, so the optimizer
            state is stored when the engine creates a checkpoint.
            Default: ``True``
        attach_handler (bool, optional): If ``True``, a handler is
            attached to the engine to consolidate the ZeRO optimizer
            state at the end of each training epoch. Consolidate the
            optimizer state dict is important to export the optimizer
            state dict. If ``False``, no handler is attached.
            Default: ``True``
    """

    def __init__(
        self,
        optimizer_config: dict,
        zero_kwargs: Optional[dict] = None,
        add_module_to_engine: bool = True,
        attach_handler: bool = True,
    ):
        self._optimizer_class = import_object(optimizer_config.pop(OBJECT_TARGET))
        self._optimizer_config = optimizer_config
        self._zero_kwargs = zero_kwargs or {}
        self._add_module_to_engine = bool(add_module_to_engine)
        self._attach_handler = bool(attach_handler)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  optimizer_class={self._optimizer_class},\n"
            f"  optimizer_config={str_indent(to_pretty_json_str(self._optimizer_config))},\n"
            f"  zero_kwargs={str_indent(to_pretty_json_str(self._zero_kwargs))},\n"
            f"  add_module_to_engine={self._add_module_to_engine},\n"
            f"  attach_handler={self._attach_handler},\n"
            ")"
        )

    def create(self, engine: BaseEngine, model: Module) -> ZeroRedundancyOptimizer:
        r"""Creates a zero redundancy optimizer (ZeRO).

        This method is responsible to register the event handlers
        associated to the optimizer.

        Args:
            engine (``gravitorch.engines.BaseEngine``): Specifies an
                engine.
            model (``torch.nn.Module``): Specifies a model.

        Returns:
            ``torch.distributed.optim.ZeroRedundancyOptimizer``: The
                created zero redundancy optimizer (ZeRO).
        """
        optimizer = ZeroRedundancyOptimizer(
            params=model.parameters(),
            optimizer_class=self._optimizer_class,
            **self._optimizer_config,
            **self._zero_kwargs,
        )
        logger.info(f"optimizer:\n{optimizer}")
        if self._add_module_to_engine:
            logger.info(f"Adding an optimizer to the engine (key: {ct.OPTIMIZER})...")
            engine.add_module(ct.OPTIMIZER, optimizer)
        if self._attach_handler:
            logger.info("Creating handler to consolidate the optimizer state dict...")
            ConsolidateOptimizerState().attach(engine)
        return optimizer
