r"""This module implements a handler to load state dict."""

__all__ = ["ModelStateDictLoader", "PartialModelStateDictLoader"]

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Optional, Union

from gravitorch.engines.base import BaseEngine
from gravitorch.engines.events import EngineEvents
from gravitorch.handlers.base import BaseHandler
from gravitorch.handlers.utils import add_unique_event_handler
from gravitorch.nn.utils.state_dict import (
    load_checkpoint_to_module,
    load_model_state_dict,
)
from gravitorch.utils.events import VanillaEventHandler
from gravitorch.utils.path import sanitize_path

logger = logging.getLogger(__name__)


class ModelStateDictLoader(BaseHandler):
    r"""Implements a handler to load the model state dict.

    Args:
        checkpoint_path (``pathlib.Path`` or str): Specifies a path
            to a model checkpoint. This weights in the checkpoint are
            used to initialize the model.
        event (str, optional): Specifies the event when to load the
            model state dict. Default: ``'started'``
        strict (bool, optional): whether to strictly enforce that the
            keys in ``state_dict`` match the keys returned by this
            module's :meth:`~torch.nn.Module.state_dict` function.
            Default: ``True``
        key (str or list or tuple or ``None``, optional): Specifies
            the key of the state dict to load. The state dict can
            contain data that are not about the module, so they may
            need to be excluded. For nested case, it is possible to
            specify the list of keys to get the good part of the state
            dict. Default: ``None``
    """

    def __init__(
        self,
        checkpoint_path: Union[Path, str],
        event: str = EngineEvents.STARTED,
        strict: bool = True,
        key: Union[str, list[str], tuple[str, ...], None] = None,
    ):
        self._checkpoint_path = sanitize_path(checkpoint_path)
        self._event = str(event)
        self._strict = bool(strict)
        self._key = key

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(checkpoint_path={self._checkpoint_path}, "
            f"event={self._event}, strict={self._strict}, key={self._key})"
        )

    def attach(self, engine: BaseEngine) -> None:
        add_unique_event_handler(
            engine=engine,
            event=self._event,
            event_handler=VanillaEventHandler(
                self.load,
                handler_kwargs={"engine": engine},
            ),
        )

    def load(self, engine: BaseEngine) -> None:
        r"""Loads a model state dict.

        Args:
            engine (``BaseEngine``): Specifies the engine with the
                model.
        """
        load_checkpoint_to_module(
            path=self._checkpoint_path,
            module=engine.model,
            strict=self._strict,
            key=self._key,
        )


class PartialModelStateDictLoader(BaseHandler):
    r"""Implements a handler to load some model weights from a checkpoint.

    Args:
        checkpoint_path (``pathlib.Path`` or str): Specifies a path
            to a model checkpoint. This weights in the checkpoint are
            used to initialize the model.
        event (str, optional): Specifies the event when to load the
            model state dict. Default: ``'started'``
        strict (bool, optional): whether to strictly enforce that the
            keys in ``state_dict`` match the keys returned by this
            module's :meth:`~torch.nn.Module.state_dict` function.
            Default: ``True``
        exclude_key_prefixes (sequence or ``None``, optional):
            Specifies the list of key prefixes to exclude when loading
            the state dict. Default: ``None``
    """

    def __init__(
        self,
        checkpoint_path: Union[Path, str],
        event: str = EngineEvents.STARTED,
        strict: bool = True,
        exclude_key_prefixes: Optional[Sequence[str]] = None,
    ):
        self._checkpoint_path = sanitize_path(checkpoint_path)
        self._event = str(event)
        self._strict = bool(strict)
        self._exclude_key_prefixes = exclude_key_prefixes or []

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(checkpoint_path={self._checkpoint_path}, event={self._event}, "
            f"strict={self._strict}, exclude_key_prefixes={self._exclude_key_prefixes})"
        )

    def attach(self, engine: BaseEngine) -> None:
        add_unique_event_handler(
            engine=engine,
            event=self._event,
            event_handler=VanillaEventHandler(
                self.load,
                handler_kwargs={"engine": engine},
            ),
        )

    def load(self, engine: BaseEngine) -> None:
        r"""Loads a model state dict.

        Args:
            engine (``BaseEngine``): Specifies the engine with the
                model.
        """
        load_model_state_dict(
            path=self._checkpoint_path,
            module=engine.model,
            strict=self._strict,
            exclude_key_prefixes=self._exclude_key_prefixes,
        )
