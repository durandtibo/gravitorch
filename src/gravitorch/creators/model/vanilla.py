__all__ = ["VanillaModelCreator"]

import logging
from typing import Union

from torch import nn

from gravitorch import constants as ct
from gravitorch.creators.model.base import BaseModelCreator
from gravitorch.engines.base import BaseEngine
from gravitorch.models.utils import setup_and_attach_model, setup_model
from gravitorch.utils.device_placement import (
    AutoDevicePlacement,
    BaseDevicePlacement,
    setup_device_placement,
)
from gravitorch.utils.format import str_indent

logger = logging.getLogger(__name__)


class VanillaModelCreator(BaseModelCreator):
    r"""Implements a vanilla model creator.

    This model creator is designed for models that run on a single
    device. If ``device_placement=True``, the device is managed by the
    function ``gravitorch.distributed.device()``.

    Args:
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
    """

    def __init__(
        self,
        model_config: dict,
        attach_model_to_engine: bool = True,
        add_module_to_engine: bool = True,
        device_placement: Union[BaseDevicePlacement, dict, None] = None,
    ):
        self._model_config = model_config
        self._attach_model_to_engine = bool(attach_model_to_engine)
        self._add_module_to_engine = bool(add_module_to_engine)
        self._device_placement = setup_device_placement(device_placement or AutoDevicePlacement())

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  attach_model_to_engine={self._attach_model_to_engine},\n"
            f"  add_module_to_engine={self._add_module_to_engine},\n"
            f"  device_placement={str_indent(self._device_placement)},\n"
            ")"
        )

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
