__all__ = [
    "EngineStateLoader",
    "EngineStateLoaderWithExcludeKeys",
    "EngineStateLoaderWithIncludeKeys",
]

import copy
import logging
from pathlib import Path
from typing import Union

import torch

from gravitorch.engines.base import BaseEngine
from gravitorch.handlers.base import BaseHandler
from gravitorch.handlers.utils import add_unique_event_handler
from gravitorch.utils.events import VanillaEventHandler
from gravitorch.utils.path import sanitize_path

logger = logging.getLogger(__name__)


class EngineStateLoader(BaseHandler):
    r"""Implements a handler to load the engine state dict from a PyTorch file.

    Note that the state dict in the file should be compatible with
    the engine.

    Args:
        path (``pathlib.Path`` or str): Specifies the path to the
            PyTorch file.
        event (str): Specifies the event used to load the engine
            state dict.
        missing_ok (bool, optional): If ``False``, a
            ``FileNotFoundError`` exception is raised if the path
            does not exist, otherwise the ``FileNotFoundError``
            exception is not raised and this handler does not try
            to load the state dict. Default: ``False``
    """

    def __init__(self, path: Union[Path, str], event: str, missing_ok: bool = False):
        self._path = sanitize_path(path)
        self._event = str(event)
        self._missing_ok = bool(missing_ok)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(event={self._event}, path={self._path}, "
            f"missing_ok={self._missing_ok})"
        )

    def attach(self, engine: BaseEngine) -> None:
        add_unique_event_handler(
            engine=engine,
            event=self._event,
            event_handler=VanillaEventHandler(
                self.load_engine_state_dict, handler_kwargs={"engine": engine}
            ),
        )

    def load_engine_state_dict(self, engine: BaseEngine) -> None:
        r"""Loads an engine state dict from a PyTorch file.

        Args:
            engine (``BaseEngine``): Specifies the engine.
        """
        if not self._path.is_file():
            if self._missing_ok:
                return
            else:
                raise FileNotFoundError(f"{self._path} is not a file")
        logger.info(f"Load state dict from {self._path}...")
        state = torch.load(self._path, map_location=lambda storage, loc: storage)
        engine.load_state_dict(self._prepare_state_dict(state))

    def _prepare_state_dict(self, state_dict: dict) -> dict:
        r"""Prepares the state dict before to load it in the engine.

        Args:
            state_dict (dict): The state dict to prepare.

        Returns:
            dict: The prepared state dict.
        """
        return state_dict


class EngineStateLoaderWithExcludeKeys(EngineStateLoader):
    r"""Implements a handler to load the engine state dict from a PyTorch file.

    This state dict loader excludes the keys that are specified in
    ``exclude_keys``. Note that the state dict in the file should be
    compatible with the engine.

    Args:
        path (``pathlib.Path`` or str): Specifies the path to the
            PyTorch file.
        event (str): Specifies the event used to load the engine
            state dict.
        exclude_keys (tuple or list): Specifies the keys to exclude
            in the state dict.
        missing_ok (bool, optional): If ``False``, a
            ``FileNotFoundError`` exception is raised if the path
            does not exist, otherwise the ``FileNotFoundError``
            exception is not raised and this handler does not try
            to load the state dict. Default: ``False``
    """

    def __init__(
        self,
        path: Union[Path, str],
        event: str,
        exclude_keys: Union[tuple[str, ...], list[str]],
        missing_ok: bool = False,
    ):
        super().__init__(path=path, event=event, missing_ok=missing_ok)
        self._exclude_keys = tuple(exclude_keys)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(event={self._event}, path={self._path}, "
            f"exclude_keys={self._exclude_keys}, missing_ok={self._missing_ok})"
        )

    def _prepare_state_dict(self, state_dict: dict) -> dict:
        r"""Prepares the state dict before to load it in the engine.

        Args:
            state_dict (dict): The state dict to prepare.

        Returns:
            dict: The prepared state dict.
        """
        new_state_dict = {}
        logger.info(
            "Before preparation, the state dict has the following keys: "
            f"{tuple(state_dict.keys())}"
        )
        for key, value in state_dict.items():
            if key not in self._exclude_keys:
                new_state_dict[key] = copy.copy(value)
        logger.info(
            "After preparation, the state dict has the following keys: "
            f"{tuple(new_state_dict.keys())}"
        )
        return new_state_dict


class EngineStateLoaderWithIncludeKeys(EngineStateLoader):
    r"""Implements a handler to load the engine state dict from a PyTorch file.

    This state dict loader loads only the keys that are specified in
    ``include_keys``. If a key is not present in the state dict, this
    key is ignored. Note that the state dict in the file should be
    compatible with the engine.

    Args:
        path (``pathlib.Path`` or str): Specifies the path to the
            PyTorch file.
        event (str): Specifies the event used to load the engine state
            dict.
        include_keys (tuple or list): Specifies the keys to include in
            the state dict. If a key is in the state dict but not in
            ``include_keys``, this key is ignored.
        missing_ok (bool, optional): If ``False``, a
            ``FileNotFoundError`` exception is raised if the path
            does not exist, otherwise the ``FileNotFoundError``
            exception is not raised and this handler does not try to
            load the state dict. Default: ``False``
    """

    def __init__(
        self,
        path: Union[Path, str],
        event: str,
        include_keys: Union[tuple[str, ...], list[str]],
        missing_ok: bool = False,
    ):
        super().__init__(path=path, event=event, missing_ok=missing_ok)
        self._include_keys = tuple(include_keys)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(event={self._event}, path={self._path}, "
            f"include_keys={self._include_keys}, missing_ok={self._missing_ok})"
        )

    def _prepare_state_dict(self, state_dict: dict) -> dict:
        r"""Prepares the state dict before to load it in the engine.

        Args:
            state_dict (dict): The state dict to prepare.

        Returns:
            dict: The prepared state dict.
        """
        new_state_dict = {}
        logger.info(
            f"Before preparation, the state dict has the following keys: {tuple(state_dict.keys())}"
        )
        for key in self._include_keys:
            if key in state_dict:
                new_state_dict[key] = copy.copy(state_dict[key])
            else:
                logger.info(f"Ignore '{key}' key because it is not present in the state dict")
        logger.info(
            "After preparation, the state dict has the following keys: "
            f"{tuple(new_state_dict.keys())}"
        )
        return new_state_dict
