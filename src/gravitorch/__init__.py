from __future__ import annotations

__all__ = ["__version__"]

from importlib.metadata import version

__version__ = version(__name__)
