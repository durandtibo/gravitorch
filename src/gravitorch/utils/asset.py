r"""This module implements an asset manager."""

from __future__ import annotations

__all__ = ["AssetExistsError", "AssetManager", "AssetNotFoundError"]

import copy
import logging
from typing import Any

from coola import objects_are_equal, summary
from coola.utils import str_indent, str_mapping

logger = logging.getLogger(__name__)


class AssetExistsError(Exception):
    r"""Raised when trying to add an asset that already exists."""


class AssetNotFoundError(Exception):
    r"""Raised when trying to access an asset that does not exist."""


class AssetManager:
    r"""Implements an asset manager.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.utils.asset import AssetManager
        >>> manager = AssetManager()
        >>> manager
        AssetManager()
        >>> manager.add_asset("mean", 5)
        >>> manager
        AssetManager(
          (mean): <class 'int'> 5
        )
        >>> manager.get_asset("mean")
        5
    """

    def __init__(self, assets: dict[str, Any] | None = None) -> None:
        self._assets = assets or {}

    def __len__(self) -> int:
        return len(self._assets)

    def __repr__(self) -> str:
        assets = {name: summary(asset) for name, asset in self._assets.items()}
        args = f"\n  {str_indent(str_mapping(assets))}\n" if assets else ""
        return f"{self.__class__.__qualname__}({args})"

    def __str__(self) -> str:
        return f"{self.__class__.__qualname__}(num_assets={len(self._assets):,})"

    def add_asset(self, name: str, asset: Any, replace_ok: bool = False) -> None:
        r"""Adds an asset to the asset manager.

        Note that the name should be unique. If the name exists, the
        old asset will be overwritten by the new asset.

        Args:
        ----
            name (str): Specifies the name of the asset to add.
            asset: Specifies the asset to add.
            replace_ok (bool, optional): If ``False``,
                ``AssetExistsError`` is raised if an asset with the
                same name exists. Default: ``False``

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.asset import AssetManager
            >>> manager = AssetManager()
            >>> manager.add_asset("mean", 5)
        """
        if name in self._assets and not replace_ok:
            raise AssetExistsError(
                f"`{name}` is already used to register an asset. "
                "Use `replace_ok=True` to replace an asset"
            )
        self._assets[name] = asset

    def clone(self) -> AssetManager:
        r"""Creates a deep copy of the current asset manager.

        Returns
        -------
            ``AssetManager``: A deep copy of the current asset manager.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.asset import AssetManager
            >>> manager = AssetManager({"name": 5})
            >>> clone = manager.clone()
            >>> manager.add_asset("name", 7, replace_ok=True)
            >>> manager
            AssetManager(
              (name): <class 'int'>  7
            )
            >>> clone
            AssetManager(
              (name): <class 'int'>  5
            )
        """
        return AssetManager(copy.deepcopy(self._assets))

    def equal(self, other: Any) -> bool:
        r"""Indicates if two objects are equal.

        Args:
        ----
            other: Specifies the object to compare with.

        Returns:
        -------
            bool: ``True`` if the two objects are equal, otherwise
                ``False``.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.asset import AssetManager
            >>> manager = AssetManager()
            >>> manager.add_asset("mean", 5)
            >>> manager2 = AssetManager()
            >>> manager.equal(manager2)
            False
            >>> manager2.add_asset("mean", 5)
            >>> manager.equal(manager2)
            True
        """
        if not isinstance(other, AssetManager):
            return False
        return objects_are_equal(self._assets, other._assets)

    def get_asset(self, name: str) -> Any:
        r"""Gets an asset.

        Args:
        ----
            name (str): Specifies the asset to get.

        Returns:
        -------
            The asset

        Raises:
        ------
            ``AssetNotFoundError`` if the asset does not exist.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.asset import AssetManager
            >>> manager = AssetManager()
            >>> manager.add_asset("mean", 5)
            >>> manager.get_asset("mean")
            5
        """
        if name not in self._assets:
            raise AssetNotFoundError(f"The asset '{name}' does not exist")
        return self._assets[name]

    def get_asset_names(self) -> tuple[str, ...]:
        r"""Gets all the asset names.

        Returns
        -------
            tuple: The asset names.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.asset import AssetManager
            >>> manager = AssetManager()
            >>> manager.add_asset("mean", 5)
            >>> manager.get_asset_names()
            ('mean',)
        """
        return tuple(self._assets.keys())

    def has_asset(self, name: str) -> bool:
        r"""Indicates if the asset exists or not.

        Args:
        ----
            name (str): Specifies the name of the asset.

        Returns:
        -------
            bool: ``True`` if the asset exists, otherwise ``False``

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.asset import AssetManager
            >>> manager = AssetManager()
            >>> manager.has_asset("mean")
            False
            >>> manager.add_asset("mean", 5)
            >>> manager.has_asset("mean")
            True
        """
        return name in self._assets

    def remove_asset(self, name: str) -> None:
        r"""Removes an asset.

        Args:
        ----
            name (str): Specifies the name of the asset to remove.

        Raises:
        ------
            ``AssetNotFoundError`` if the asset does not exist.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.asset import AssetManager
            >>> manager = AssetManager()
            >>> manager.add_asset("mean", 5)
            >>> manager.remove_asset("mean")
            >>> manager.has_asset("mean")
            False
        """
        if name not in self._assets:
            raise AssetNotFoundError(
                f"The asset '{name}' does not exist so it is not possible to remove it"
            )
        del self._assets[name]

    def load_state_dict(self, state_dict: dict, keys: list | tuple | None = None) -> None:
        r"""Loads the state dict of each module.

        Note this method ignore the missing modules or keys. For
        example if you want to load the optimizer module but there is
        no 'optimizer' key in the state dict, this method will ignore
        the optimizer module.

        Args:
        ----
            state_dict (dict): Specifies the state dict to load.
            keys (list or tuple or ``None``): Specifies the keys to
                load. If ``None``, it loads all the keys associated
                to the registered modules.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from torch import nn
            >>> from gravitorch.utils.asset import AssetManager
            >>> manager = AssetManager()
            >>> manager.add_asset("my_module", nn.Linear(4, 6))
            >>> manager.load_state_dict(
            ...     {"my_module": {"weight": torch.ones(6, 4), "bias": torch.zeros(6)}}
            ... )
        """
        keys = keys or tuple(self._assets.keys())
        for key in keys:
            if key not in state_dict:
                logger.info(f"Ignore key {key} because it is not in the state dict")
                continue
            if key not in self._assets:
                logger.info(f"Ignore key {key} because there is no module associated to it")
                continue
            if not hasattr(self._assets[key], "load_state_dict"):
                logger.info(
                    f"Ignore key {key} because the module does not have 'load_state_dict' method"
                )
                continue
            self._assets[key].load_state_dict(state_dict[key])

    def state_dict(self) -> dict:
        r"""Creates a state dict with all the modules.

        The state of each module is store with the associated key of
        the module.

        Returns
        -------
            dict: The state dict of all the modules.

        Example usage:

        .. code-block:: pycon

            >>> from torch import nn
            >>> from gravitorch.utils.asset import AssetManager
            >>> manager = AssetManager()
            >>> manager.add_asset("my_module", nn.Linear(4, 6))
            >>> manager.state_dict()
            {'my_module': OrderedDict([('weight', tensor([[...)), ('bias', tensor([...))])}
            >>> manager.add_asset("int", 123)
            >>> manager.state_dict()
            {'my_module': OrderedDict([('weight', tensor([[...]])), ('bias', tensor([...]))])}
        """
        state = {}
        for name, module in self._assets.items():
            if hasattr(module, "state_dict"):
                state[name] = module.state_dict()
            else:
                logger.info(f"Skip '{name}' module because it does not have 'state_dict' method")
        return state
