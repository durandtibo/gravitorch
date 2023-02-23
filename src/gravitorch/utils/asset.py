r"""This module implements an asset manager."""

__all__ = ["AssetExistsError", "AssetManager", "AssetNotFoundError", "get_asset_summary"]

import copy
import logging
from typing import Any, Optional

import numpy as np
import torch
from coola import objects_are_equal

from gravitorch.utils.format import str_indent, to_torch_mapping_str

logger = logging.getLogger(__name__)


class AssetExistsError(Exception):
    r"""Raised when trying to add an asset that already exists."""


class AssetNotFoundError(Exception):
    r"""Raised when trying to access an asset that does not exist."""


class AssetManager:
    r"""Implements an asset manager."""

    def __init__(self, assets: Optional[dict[str, Any]] = None):
        self._assets = assets or {}

    def __repr__(self) -> str:
        summaries = {name: self._get_asset_summary(asset) for name, asset in self._assets.items()}
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  {str_indent(to_torch_mapping_str(summaries))}\n"
            ")"
        )

    def __str__(self) -> str:
        return f"{self.__class__.__qualname__}(num_assets={len(self._assets):,})"

    def add_asset(self, name: str, asset: Any, replace_ok: bool = False) -> None:
        r"""Adds an asset to the asset manager.

        Note that the name should be unique. If the name exists, the
        old asset will be overwritten by the new asset.

        Args:
            name (str): Specifies the name of the asset to add.
            asset: Specifies the asset to add.
            replace_ok (bool, optional): If ``False``,
                ``AssetExistsError`` is raised if an asset with the
                same name exists. Default: ``False``

        Example usage:

        .. code-block:: python

            >>> from gravitorch.utils.asset import AssetManager
            >>> manager = AssetManager()
            >>> manager.add_asset('mean', 5)
        """
        if name in self._assets and not replace_ok:
            raise AssetExistsError(
                f"{name} is already used to register an asset. "
                "Use `replace_ok=True` to replace an asset"
            )
        self._assets[name] = asset

    def clone(self) -> "AssetManager":
        r"""Creates a deep copy of the current asset manager.

        Returns:
            ``AssetManager``: A deep copy of the current asset manager.

        Example usage:

        .. code-block:: python

            >>> from gravitorch.utils.asset import AssetManager
            >>> manager = AssetManager({'name': 5})
            >>> clone = manager.clone()
            >>> manager.add_asset('name', 7)
            >>> manager
            AssetManager(
              (name) <class 'int'>  7
            )
            >>> clone
            AssetManager(
              (name) <class 'int'>  5
            )
        """
        return AssetManager(copy.deepcopy(self._assets))

    def equal(self, other) -> bool:
        r"""Indicates if two objects are equal.

        Args:
            other: Specifies the object to compare with.

        Returns:
            bool: ``True`` if the two objects are equal, otherwise
                ``False``.

        Example usage:

        .. code-block:: python

            >>> from gravitorch.utils.asset import AssetManager
            >>> manager = AssetManager()
            >>> manager.add_asset('mean', 5)
            >>> manager2 = AssetManager()
            >>> manager.equal(manager2)
            False
            >>> manager2.add_asset('mean', 5)
            >>> manager.equal(manager2)
            True
        """
        if not isinstance(other, AssetManager):
            return False
        return objects_are_equal(self._assets, other._assets)

    def get_asset(self, name: str) -> Any:
        r"""Gets an asset.

        Args:
            name (str): Specifies the asset to get.

        Returns:
            The asset

        Raises:
            ``AssetNotFoundError`` if the asset does not exist.

        Example usage:

        .. code-block:: python

            >>> from gravitorch.utils.asset import AssetManager
            >>> manager = AssetManager()
            >>> manager.add_asset('mean', 5)
            >>> manager.get_asset('mean')
            5
        """
        if name not in self._assets:
            raise AssetNotFoundError(f"The asset {name} does not exist")
        return self._assets[name]

    def get_asset_names(self) -> tuple[str, ...]:
        r"""Gets all the asset names.

        Returns:
            tuple: The asset names.

        Example usage:

        .. code-block:: python

            >>> from gravitorch.utils.asset import AssetManager
            >>> manager = AssetManager()
            >>> manager.add_asset('mean', 5)
            >>> manager.get_asset_names()
            ('mean',)
        """
        return tuple(self._assets.keys())

    def has_asset(self, name: str) -> bool:
        r"""Indicates if the asset exists or not.

        Args:
            name (str): Specifies the name of the asset.

        Returns:
            bool: ``True`` if the asset exists, otherwise ``False``

        Example usage:

        .. code-block:: python

            >>> from gravitorch.utils.asset import AssetManager
            >>> manager = AssetManager()
            >>> manager.has_asset('mean')
            False
            >>> manager.add_asset('mean', 5)
            >>> manager.has_asset('mean')
            True
        """
        return name in self._assets

    def remove_asset(self, name: str) -> None:
        r"""Removes an asset.

        Args:
            name (str): Specifies the name of the asset to remove.

        Raises:
            ``AssetNotFoundError`` if the asset does not exist.

        Example usage:

        .. code-block:: python

            >>> from gravitorch.utils.asset import AssetManager
            >>> manager = AssetManager()
            >>> manager.add_asset('mean', 5)
            >>> manager.remove_asset('mean')
            5
            >>> manager.has_asset('mean')
            False
        """
        if name not in self._assets:
            raise AssetNotFoundError(
                f"The asset {name} does not exist so it is not possible to remove it"
            )
        del self._assets[name]

    def _get_asset_summary(self, asset: Any) -> str:
        return get_asset_summary(asset)


def get_asset_summary(asset: Any) -> str:
    r"""Gets a one line summary of an asset.

    Args:
        asset: Specifies the asset.

    Returns:
        str: A one line summary of an asset.

    Example usage:

    .. code-block:: python

        >>> import torch
        >>> from gravitorch.utils.asset import get_asset_summary
        >>> get_asset_summary(torch.ones(2, 3))
        <class 'torch.Tensor'>  shape=torch.Size([2, 3])  dtype=torch.float32  device=cpu
    """
    if torch.is_tensor(asset):
        return f"{type(asset)}  shape={asset.shape}  dtype={asset.dtype}  device={asset.device}"
    if isinstance(asset, np.ndarray):
        return f"{type(asset)}  shape={asset.shape}  dtype={asset.dtype}"
    if isinstance(asset, (bool, int, float)):
        return f"{type(asset)}  {asset}"
    return f"{type(asset)}"
