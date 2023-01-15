__all__ = ["get_best_values", "get_last_values"]

from collections.abc import Mapping
from typing import Any

from gravitorch.utils.history.base import BaseHistory, EmptyHistoryError


def get_best_values(
    histories: Mapping[str, BaseHistory], prefix: str = "", suffix: str = ""
) -> dict[str, Any]:
    """Gets the best value of each history.

    This function ignores the empty histories and non-comparable
    histories.

    Args:
        histories (``Mapping``): Specifies the histories and their
            keys.
        prefix (str, optional): Specifies the prefix used to create
            the dict of best values. The goal of this prefix is to
            generate a name which is different from the history name
            to avoid confusion. By default, the returned dict uses
            the same name as the history. Default: ``''``
        suffix (str, optional): Specifies the suffix used to create
            the dict of best values. The goal of this suffix is to
            generate a name which is different from the history name
            to avoid confusion. By default, the returned dict uses
            the same name as the history. Default: ``''``

    Returns:
        dict: The dict with the best value of each history.

    Example:

    .. code-block:: python

        >>> from gravitorch.utils.history import (
        ...     MinScalarHistory,
        ...     MaxScalarHistory,
        ...     get_best_values,
        ... )
        >>> history1 = MinScalarHistory('loss')
        >>> history1.add_value(1.9)
        >>> history1.add_value(1.2)
        >>> history2 = MaxScalarHistory('accuracy')
        >>> history2.add_value(42)
        >>> history2.add_value(35)
        >>> get_best_values({'loss': history1, 'accuracy': history2})
        {'loss': 1.2, 'accuracy': 42}
        >>> get_best_values({'loss': history1, 'accuracy': history2}, prefix='best/')
        {'best/loss': 1.2, 'best/accuracy': 42}
        >>> get_best_values({'loss': history1, 'accuracy': history2}, suffix='/best')
        {'loss/best': 1.2, 'accuracy/best': 42}
    """
    values = {}
    for key, history in histories.items():
        if history.is_comparable():
            try:
                values[f"{prefix}{key}{suffix}"] = history.get_best_value()
            except EmptyHistoryError:
                pass
    return values


def get_last_values(
    histories: Mapping[str, BaseHistory], prefix: str = "", suffix: str = ""
) -> dict[str, Any]:
    """Gets the last value of each history.

    This function ignores the empty histories.

    Args:
        histories (``Mapping``): Specifies the histories and their
            keys.
        prefix (str, optional): Specifies the prefix used to create
            the dict of best values. The goal of this prefix is to
            generate a name which is different from the hisotry name
            to avoid confusion. By default, the returned dict uses
            the same name as the history. Default: ``''``
        suffix (str, optional): Specifies the suffix used to create
            the dict of best values. The goal of this suffix is to
            generate a name which is different from the history name
            to avoid confusion. By default, the returned dict uses
            the same name as the history. Default: ``''``

    Returns:
        dict: The dict with the best value of each history.

    Example:

    .. code-block:: python

        >>> from gravitorch.utils.history import (
        ...     MinScalarHistory,
        ...     MaxScalarHistory,
        ...     get_best_values,
        ... )
        >>> history1 = MinScalarHistory('loss')
        >>> history1.add_value(1.9)
        >>> history1.add_value(1.2)
        >>> history2 = MaxScalarHistory('accuracy')
        >>> history2.add_value(42)
        >>> history2.add_value(35)
        >>> get_best_values({'loss': history1, 'accuracy': history2})
        {'loss': 1.2, 'accuracy': 42}
        >>> get_best_values({'loss': history1, 'accuracy': history2}, prefix='best/')
        {'best/loss': 1.2, 'best/accuracy': 42}
        >>> get_best_values({'loss': history1, 'accuracy': history2}, suffix='/best')
        {'loss/best': 1.2, 'accuracy/best': 42}
    """
    values = {}
    for key, history in histories.items():
        try:
            values[f"{prefix}{key}{suffix}"] = history.get_last_value()
        except EmptyHistoryError:
            pass
    return values
