r"""This module implements an asset manager."""

__all__ = ["AssetManager", "AssetNotFoundError"]

import logging
from typing import Any

logger = logging.getLogger(__name__)


class AssetManager:
    r"""Implements an asset manager."""

    def __init__(self):
        self._assets = {}

    def __str__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def add_asset(self, name: str, asset: Any) -> None:
        r"""Adds an asset to the asset manager.

        Note that the name should be unique. If the name exists, the
        old asset will be overwritten by the new asset.

        Args:
            name (str): Specifies the name of the asset to add.
            asset: Specifies the asset to add.

        Example usage:

        .. code-block:: python

            >>> from gravitorch.utils import AssetManager
            >>> manager = AssetManager()
            >>> manager.add_asset('mean', 5)
        """
        self._assets[name] = asset

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

            >>> from gravitorch.utils import AssetManager
            >>> manager = AssetManager()
            >>> manager.add_asset('mean', 5)
            >>> manager.get_asset('mean')
            5
        """
        if name not in self._assets:
            raise AssetNotFoundError(f"The asset {name} does not exist")
        return self._assets[name]

    def has_asset(self, name: str) -> bool:
        r"""Indicates if the asset exists or not.

        Args:
            name (str): Specifies the name of the asset.

        Returns:
            bool: ``True`` if the asset exists, otherwise ``False``

        Example usage:

        .. code-block:: python

            >>> from gravitorch.utils import AssetManager
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

            >>> from gravitorch.utils import AssetManager
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


class AssetNotFoundError(Exception):
    r"""Raised when trying to access an asset that does not exist."""
