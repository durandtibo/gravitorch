from __future__ import annotations

__all__ = ["DataDistributedParallelModelCreator", "to_ddp"]

import logging
from typing import TYPE_CHECKING

from coola.utils import str_indent
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from gravitorch.creators.model.base import BaseModelCreator, setup_model_creator
from gravitorch.distributed import comm as dist
from gravitorch.utils.format import str_pretty_json

if TYPE_CHECKING:
    from gravitorch.engines import BaseEngine

logger = logging.getLogger(__name__)


class DataDistributedParallelModelCreator(BaseModelCreator):
    r"""Implements a model creator that wraps a created model with
    ``DistributedDataParallel``.

    Args:
    ----
        model_creator (``BaseModelCreator`` or dict): Specifies a
            model creator or its configuration. The created model
            should be compatible with ``DistributedDataParallel``.
        ddp_kwargs (dict or ``None``): Specifies some keyword
            arguments used to instantiate the
            ``DistributedDataParallel``. Please read the documentation
            of ``DistributedDataParallel`` to see the possible
            options. Note that it is not possible to set ``module``
            and ``device_ids`` with a keyword argument.
            Default: ``None``
    """

    def __init__(
        self, model_creator: BaseModelCreator | dict, ddp_kwargs: dict | None = None
    ) -> None:
        self._model_creator = setup_model_creator(model_creator)
        self._ddp_kwargs = ddp_kwargs or {}

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  model_creator={self._model_creator},\n"
            f"  ddp_kwargs={str_indent(str_pretty_json(self._ddp_kwargs))},\n"
            ")"
        )

    def create(self, engine: BaseEngine) -> nn.Module:
        model = self._model_creator.create(engine)
        return to_ddp(module=model, ddp_kwargs=self._ddp_kwargs)


def to_ddp(module: nn.Module, ddp_kwargs: dict | None = None) -> nn.Module:
    r"""Wraps a module with the ``DistributedDataParallel`` module.

    Args:
    ----
        module (``torch.nn.Module``): Specifies the module to wrap
            with ``DistributedDataParallel``. The module should be
            compatible with ``DistributedDataParallel``. If you use
            NCCL, the module should be on a CUDA device.
        ddp_kwargs (dict or ``None``): Specifies some keyword
            arguments used to instantiate the
            ``DistributedDataParallel``. Please read the
            documentation of ``DistributedDataParallel`` to see the
            possible options. Note that it is not possible to set
            ``module`` and ``device_ids`` with a keyword argument.
            Default: ``None``

    Returns:
    -------
        ``torch.nn.Module``: The model wrapped in a
            ``DistributedDataParallel`` module.
    """
    if isinstance(module, DistributedDataParallel):
        logger.warning(
            "No operation is performed because the module is already a DistributedDataParallel"
        )
        return module
    ddp_kwargs = ddp_kwargs or {}
    backend = dist.backend()
    if backend == dist.Backend.NCCL:
        lrank = dist.get_local_rank()
        logger.info(f"Applying DistributedDataParallel on module, device id: {lrank}")
        return DistributedDataParallel(module, device_ids=[lrank], **ddp_kwargs)
    if backend == dist.Backend.GLOO:
        logger.info("Applying DistributedDataParallel on module")
        return DistributedDataParallel(module, **ddp_kwargs)
    return module
