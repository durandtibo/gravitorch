from __future__ import annotations

__all__ = ["EmptyMeterError"]


class EmptyMeterError(Exception):
    r"""Generates an error if the metric is empty because it is not
    possible to evaluate an empty metric."""
