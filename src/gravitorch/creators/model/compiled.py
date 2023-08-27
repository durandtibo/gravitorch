from __future__ import annotations

__all__ = ["CompiledModelCreator"]

import logging

import torch
from coola.utils import str_indent
from torch.nn import Module

from gravitorch.creators.model.base import BaseModelCreator, setup_model_creator
from gravitorch.engines.base import BaseEngine
from gravitorch.utils.format import str_pretty_json

logger = logging.getLogger(__name__)


class CompiledModelCreator(BaseModelCreator):
    r"""Implements a model creator that compiles a model with
    ``torch.compile``.

    Args:
    ----
        model_creator (``BaseModelCreator`` or dict): Specifies a
            model creator or its configuration. The created model
            should be compatible with ``DistributedDataParallel``.
        compile_kwargs (dict or ``None``): Specifies some keyword
            arguments used to compile the model. Please read the
            documentation of ``torch.compile`` to see the possible
            options. Default: ``None``
    """

    def __init__(
        self,
        model_creator: BaseModelCreator | dict,
        compile_kwargs: dict | None = None,
    ) -> None:
        self._model_creator = setup_model_creator(model_creator)
        self._compile_kwargs = compile_kwargs or {}

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  model_creator={self._model_creator},\n"
            f"  compile_kwargs={str_indent(str_pretty_json(self._compile_kwargs))},\n"
            ")"
        )

    def create(self, engine: BaseEngine) -> Module:
        return torch.compile(self._model_creator.create(engine), **self._compile_kwargs)
