from __future__ import annotations

__all__ = ["CompiledModelCreator"]

import logging
from typing import TYPE_CHECKING

import torch
from coola.utils import str_indent, str_mapping
from torch.nn import Module

from gravitorch.creators.model.base import BaseModelCreator, setup_model_creator
from gravitorch.utils.format import str_pretty_json

if TYPE_CHECKING:
    from gravitorch.engines import BaseEngine

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

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.testing import create_dummy_engine
        >>> from gravitorch.creators.model import (
        ...     CompiledModelCreator,
        ...     ModelCreator,
        ... )
        >>> creator = CompiledModelCreator(
        ...     ModelCreator({"_target_": "gravitorch.testing.DummyClassificationModel"})
        ... )
        >>> creator
        CompiledModelCreator(
          (model_creator): ModelCreator(
              (model_config): {'_target_': 'gravitorch.testing.DummyClassificationModel'}
              (attach_model_to_engine): True
              (add_module_to_engine): True
              (device_placement): AutoDevicePlacement(device=cpu)
            )
          (compile_kwargs): {}
        )
        >>> engine = create_dummy_engine()
        >>> model = creator.create(engine)
        >>> model
        OptimizedModule(
          (_orig_mod): DummyClassificationModel(
            (linear): Linear(in_features=4, out_features=3, bias=True)
            (criterion): CrossEntropyLoss()
          )
        )
    """

    def __init__(
        self,
        model_creator: BaseModelCreator | dict,
        compile_kwargs: dict | None = None,
    ) -> None:
        self._model_creator = setup_model_creator(model_creator)
        self._compile_kwargs = compile_kwargs or {}

    def __repr__(self) -> str:
        args = str_indent(
            str_mapping(
                {
                    "model_creator": self._model_creator,
                    "compile_kwargs": str_pretty_json(self._compile_kwargs),
                }
            )
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def create(self, engine: BaseEngine) -> Module:
        return torch.compile(self._model_creator.create(engine), **self._compile_kwargs)
