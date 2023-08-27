r"""This package contains the optimizer base class and some implemented
optimizers."""

from __future__ import annotations

__all__ = ["NoOpOptimizer", "setup_optimizer"]

from gravitorch.optimizers.factory import setup_optimizer
from gravitorch.optimizers.noop import NoOpOptimizer
