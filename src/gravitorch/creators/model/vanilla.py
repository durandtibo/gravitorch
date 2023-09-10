from __future__ import annotations

__all__ = ["ModelCreator"]

import logging
from typing import TYPE_CHECKING

from coola.utils import str_indent, str_mapping
from torch import nn

from gravitorch import constants as ct
from gravitorch.creators.model.base import BaseModelCreator
from gravitorch.models.utils import setup_and_attach_model, setup_model
from gravitorch.utils.device_placement import (
    AutoDevicePlacement,
    BaseDevicePlacement,
    setup_device_placement,
)
from gravitorch.utils.format import str_pretty_json

if TYPE_CHECKING:
    from gravitorch.engines import BaseEngine

logger = logging.getLogger(__name__)


class ModelCreator(BaseModelCreator):
    r"""Implements a vanilla model creator.

    This model creator is designed for models that run on a single
    device. If ``device_placement=True``, the device is managed by the
    function ``gravitorch.distributed.device()``.

    Args:
    ----
        model_config (dict): Specifies the model configuration.
        attach_model_to_engine (bool, optional): If ``True``, the
            model is attached to the engine. Default: ``True``
        add_module_to_engine (bool, optional): If ``True``, the model
            is added to the engine state, so the model state is stored
            when the engine creates a checkpoint. Default: ``True``
        device_placement (bool, optional): Specifies the device
            placement module. This module moves the model on a target
            device. If ``None``, an ``AutoDevicePlacement`` object is
            instantiated. Default: ``None``

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.testing import create_dummy_engine
        >>> from gravitorch.creators.model import ModelCreator
        >>> creator = ModelCreator({"_target_": "gravitorch.testing.DummyClassificationModel"})
        >>> creator
        ModelCreator(
          (model_config): {'_target_': 'gravitorch.testing.DummyClassificationModel'}
          (attach_model_to_engine): True
          (add_module_to_engine): True
          (device_placement): AutoDevicePlacement(device=cpu)
        )
        >>> engine = create_dummy_engine()
        >>> model = creator.create(engine)
        >>> model
        DummyClassificationModel(
          (linear): Linear(in_features=4, out_features=3, bias=True)
          (criterion): CrossEntropyLoss()
        )
    """

    def __init__(
        self,
        model_config: dict,
        attach_model_to_engine: bool = True,
        add_module_to_engine: bool = True,
        device_placement: BaseDevicePlacement | dict | None = None,
    ) -> None:
        self._model_config = model_config
        self._attach_model_to_engine = bool(attach_model_to_engine)
        self._add_module_to_engine = bool(add_module_to_engine)
        self._device_placement = setup_device_placement(device_placement or AutoDevicePlacement())

    def __repr__(self) -> str:
        args = str_indent(
            str_mapping(
                {
                    "model_config": str_pretty_json(self._model_config),
                    "attach_model_to_engine": self._attach_model_to_engine,
                    "add_module_to_engine": self._add_module_to_engine,
                    "device_placement": self._device_placement,
                }
            )
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def create(self, engine: BaseEngine) -> nn.Module:
        logger.info("Creating model...")
        if self._attach_model_to_engine:
            model = setup_and_attach_model(engine=engine, model=self._model_config)
        else:
            model = setup_model(model=self._model_config)
        model = self._device_placement.send(model)
        if self._add_module_to_engine:
            logger.info(f"Adding a model to the engine state (key: {ct.MODEL})...")
            engine.add_module(ct.MODEL, model)
        return model
