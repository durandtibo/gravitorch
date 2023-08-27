from __future__ import annotations

__all__ = ["ScalarMeter"]

from collections import deque
from collections.abc import Iterable
from typing import Any

import torch

from gravitorch.utils.format import str_pretty_dict
from gravitorch.utils.meters.exceptions import EmptyMeterError


class ScalarMeter:
    r"""Implementation of a scalar meter.

    This meter tracks the following values:

        - the sum of the values
        - the number of values
        - the minimum value
        - the maximum value
        - the last N values which are used to compute the median value.

    Args:
    ----
        total (float, optional): Specifies the initial total value.
            Default: ``0.0``
        count (int, optional): Specifies the initial count value.
            Default: ``0``
        min_value (float, optional): Specifies the initial minimum
            value. Default: ``inf``
        max_value (float, optional): Specifies the initial maximum
            value. Default: ``-inf``
        values (iterable, optional): Specifies the initial last
            values to store.
        max_size (int, optional): Specifies the maximum size used to
            store the last values because it may not be possible to
            store all the values. This parameter is used to compute
            the median only on the N last values. Default: ``100``

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.utils.meters import ScalarMeter
        >>> meter = ScalarMeter()
        >>> meter.update(6)
        >>> meter.update_sequence([1, 2, 3, 4, 5, 0])
        >>> print(meter)  # xdoctest: +ELLIPSIS
        ScalarMeter
          average : 3.0
          count   : 7
          max     : 6.0
          median  : 3.0
          min     : 0.0
          std     : 2.16024684906...
          sum     : 21.0
        >>> meter.average()
        3.0
    """

    def __init__(
        self,
        total: float = 0.0,
        count: int = 0,
        min_value: float = float("inf"),
        max_value: float = -float("inf"),
        values: Iterable[float] = (),
        max_size: int = 100,
    ) -> None:
        self._total = float(total)
        self._count = int(count)
        self._min_value = float(min_value)
        self._max_value = float(max_value)
        # Store only the N last values to scale to large number of values.
        self._values = deque(values, maxlen=max_size)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(count={self._count:,}, total={self._total}, "
            f"min_value={self._min_value}, max_value={self._max_value}, "
            f"max_size={self._values.maxlen:,})"
        )

    def __str__(self) -> str:
        count = self.count
        stats = str_pretty_dict(
            {
                "average": self.average() if count else "N/A (empty)",
                "count": count,
                "max": self.max() if count else "N/A (empty)",
                "median": self.median() if count else "N/A (empty)",
                "min": self.min() if count else "N/A (empty)",
                "std": self.std() if count else "N/A (empty)",
                "sum": self.sum() if count else "N/A (empty)",
            },
            indent=2,
        )
        return f"{self.__class__.__qualname__}\n{stats}"

    @property
    def count(self) -> int:
        r"""``int``: The number of examples in the meter."""
        return self._count

    @property
    def total(self) -> float:
        r"""``float``: The total of the values added to the meter."""
        return self._total

    @property
    def values(self) -> tuple[float, ...]:
        r"""``tuple``: The values store in this meter.

        If there are more values that the maximum size, only the last
        values are returned.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import ScalarMeter
            >>> meter = ScalarMeter()
            >>> meter.update_sequence([1, 2, 3, 4, 5, 0])
            >>> meter.values
            (1.0, 2.0, 3.0, 4.0, 5.0, 0.0)
        """
        return tuple(self._values)

    def average(self) -> float:
        r"""Computes the average value.

        Returns
        -------
            float: The average value.

        Raises
        ------
            ``EmptyMeterError`` if the meter is empty.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import ScalarMeter
            >>> meter = ScalarMeter()
            >>> meter.update_sequence([1, 2, 3, 4, 5, 0])
            >>> meter.average()
            2.5
        """
        if not self._count:
            raise EmptyMeterError("The meter is empty")
        return self._total / float(self._count)

    def equal(self, other: Any) -> bool:
        r"""Indicates if two meters are equal or not.

        Args:
        ----
            other: Specifies the value to compare.

        Returns:
        -------
            bool: ``True`` if the meters are equal,
                ``False`` otherwise.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from gravitorch.utils.meters import ScalarMeter
            >>> meter1 = ScalarMeter()
            >>> meter1.update_sequence([1, 2, 3, 4, 5, 0])
            >>> meter2 = ScalarMeter()
            >>> meter2.update_sequence([1, 1, 1])
            >>> meter1.equal(meter2)
            False
        """
        if not isinstance(other, ScalarMeter):
            return False
        return self.state_dict() == other.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        r"""Loads a state to the history tracker.

        Args:
        ----
            state_dict (dict): Dictionary containing state keys with
                values.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from gravitorch.utils.meters import ScalarMeter
            >>> meter = ScalarMeter()
            >>> meter.load_state_dict(
            ...     {
            ...         "count": 6,
            ...         "total": 15.0,
            ...         "values": (1.0, 2.0, 3.0, 4.0, 5.0, 0.0),
            ...         "max_value": 5.0,
            ...         "min_value": 0.0,
            ...     }
            ... )
            >>> meter.count
            6
            >>> meter.min()
            0.0
            >>> meter.max()
            5.0
            >>> meter.sum()
            15.0
        """
        self._total = state_dict["total"]
        self._count = state_dict["count"]
        self._max_value = state_dict["max_value"]
        self._min_value = state_dict["min_value"]
        self._values.clear()
        self._values.extend(state_dict["values"])

    def max(self) -> float:
        r"""Gets the max value.

        Returns
        -------
            float: The max value.

        Raises
        ------
            ``EmptyMeterError`` if the meter is empty.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import ScalarMeter
            >>> meter = ScalarMeter()
            >>> meter.update_sequence([1, 2, 3, 4, 5, 0])
            >>> meter.max()
            5.0
        """
        if not self._count:
            raise EmptyMeterError("The meter is empty")
        return self._max_value

    def median(self) -> float:
        r"""Computes the median value from the last examples.

        If there are more values than the maximum window size, only
        the last examples are used. Internally, this meter uses a
        deque to track the last values and the median value is
        computed on the values in the deque. The median is not unique
        for input tensors with an even number of elements. In this
        case the lower of the two medians is returned.

        Returns
        -------
            float: The median value from the last examples.

        Raises
        ------
            ``EmptyMeterError`` if the meter is empty.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import ScalarMeter
            >>> meter = ScalarMeter()
            >>> meter.update_sequence([1, 2, 3, 4, 5])
            >>> meter.average()
            3.0
        """
        if not self._count:
            raise EmptyMeterError("The meter is empty")
        return torch.as_tensor(list(self._values)).median().item()

    def merge(self, meters: Iterable[ScalarMeter]) -> ScalarMeter:
        r"""Merges several meters with the current meter and returns a
        new meter.

        Only the values of the current meter are copied to the merged
        meter.

        Args:
        ----
            meters (iterable): Specifies the meters to merge to the
                current meter.

        Returns:
        -------
            ``ScalarMeter``: The merged meter.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from gravitorch.utils.meters import ScalarMeter
            >>> meter1 = ScalarMeter()
            >>> meter1.update_sequence((4, 5, 6, 7, 8, 3))
            >>> meter2 = ScalarMeter()
            >>> meter2.update_sequence([1, 1, 1])
            >>> meter3 = meter1.merge([meter2])
            >>> meter3.count
            9
            >>> meter3.max()
            8.0
            >>> meter3.min()
            1.0
            >>> meter3.sum()
            36.0
        """
        count, total = self._count, self._total
        min_value, max_value = self._min_value, self._max_value
        for meter in meters:
            count += meter.count
            total += meter.total
            min_value = min(min_value, meter._min_value)
            max_value = max(max_value, meter._max_value)
        return ScalarMeter(
            total=total,
            count=count,
            min_value=min_value,
            max_value=max_value,
            values=self.values,
            max_size=self._values.maxlen,
        )

    def merge_(self, meters: Iterable[ScalarMeter]) -> None:
        r"""Merges several meters into the current meter.

        In-place version of ``merge``.

        Only the values of the current meter are copied to the merged
        meter.

        Args:
        ----
            meters (iterable): Specifies the meters to merge to the
                current meter.

        Returns:
        -------
            ``ScalarMeter``: The merged meter.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from gravitorch.utils.meters import ScalarMeter
            >>> meter1 = ScalarMeter()
            >>> meter1.update_sequence((4, 5, 6, 7, 8, 3))
            >>> meter2 = ScalarMeter()
            >>> meter2.update_sequence([1, 1, 1])
            >>> meter1.merge_([meter2])
            >>> meter1.count
            9
            >>> meter1.max()
            8.0
            >>> meter1.min()
            1.0
            >>> meter1.sum()
            36.0
        """
        for meter in meters:
            self._count += meter.count
            self._total += meter.total
            self._min_value = min(self._min_value, meter._min_value)
            self._max_value = max(self._max_value, meter._max_value)

    def min(self) -> float:
        r"""Gets the min value.

        Returns
        -------
            float: The min value.

        Raises
        ------
            ``EmptyMeterError`` if the meter is empty.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import ScalarMeter
            >>> meter = ScalarMeter()
            >>> meter.update_sequence([1, 2, 3, 4, 5, 0])
            >>> meter.min()
            0.0
        """
        if not self._count:
            raise EmptyMeterError("The meter is empty")
        return self._min_value

    def reset(self) -> None:
        r"""Resets the meter.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from gravitorch.utils.meters import ScalarMeter
            >>> meter = ScalarMeter()
            >>> meter.update_sequence([1, 2, 3, 4, 5, 0])
            >>> meter.reset()
            >>> meter.count
            0
            >>> meter.total
            0.0
        """
        self._total = 0.0
        self._count = 0
        self._min_value = float("inf")
        self._max_value = -float("inf")
        self._values.clear()

    def state_dict(self) -> dict[str, Any]:
        r"""Returns a dictionary containing state values.

        Returns
        -------
            dict: The state values in a dict.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from gravitorch.utils.meters import ScalarMeter
            >>> meter = ScalarMeter()
            >>> meter.update_sequence([1, 2, 3, 4, 5, 0])
            >>> meter.state_dict()
            {'count': 6, 'total': 15.0, 'values': (1.0, 2.0, 3.0, 4.0, 5.0, 0.0), 'max_value': 5.0, 'min_value': 0.0}
        """
        return {
            "count": self._count,
            "total": self._total,
            "values": tuple(self._values),
            "max_value": self._max_value,
            "min_value": self._min_value,
        }

    def std(self) -> float:
        r"""Computes the standard deviation from the last examples.

        If there are more values than the maximum window size, only
        the last examples are used. Internally, this meter uses a
        deque to track the last values and the standard deviation
        is computed on the values in the deque.

        Returns
        -------
            float: The standard deviation from the last examples.

        Raises
        ------
            ``EmptyMeterError`` if the meter is empty.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import ScalarMeter
            >>> meter = ScalarMeter()
            >>> meter.update_sequence([1, 2, 3, 4, 5, 0])
            >>> meter.std()  # xdoctest: +ELLIPSIS
            1.8708287477...
        """
        if not self._count:
            raise EmptyMeterError("The meter is empty")
        return torch.as_tensor(self.values, dtype=torch.float).std(dim=0).item()

    def sum(self) -> float:
        r"""Computes the sum value.

        Returns
        -------
            float: The sum value.

        Raises
        ------
            ``EmptyMeterError`` if the meter is empty.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import ScalarMeter
            >>> meter = ScalarMeter()
            >>> meter.update_sequence([1, 2, 3, 4, 5, 0])
            >>> meter.sum()
            15.0
        """
        if not self._count:
            raise EmptyMeterError("The meter is empty")
        return self._total

    def update(self, value: float) -> None:
        r"""Updates the meter given a new value.

        Args:
        ----
            value (float): Specifies the value to add to the meter.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import ScalarMeter
            >>> meter = ScalarMeter()
            >>> meter.update(6)
            >>> meter.count
            1
            >>> meter.sum()
            6.0
        """
        value = float(value)
        self._total += value
        self._count += 1
        self._min_value = min(self._min_value, value)
        self._max_value = max(self._max_value, value)
        self._values.append(value)

    def update_sequence(self, values: list[float] | tuple[float, ...]) -> None:
        r"""Updates the meter given a list/tuple of values.

        Args:
        ----
            values (list or tuple): Specifies the list/tuple of
                values to add to the meter.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import ScalarMeter
            >>> meter = ScalarMeter()
            >>> meter.update_sequence([1, 2, 3, 4, 5, 0])
            >>> meter.count
            6
        """
        self._total += float(sum(values))
        self._count += len(values)
        self._min_value = float(min(self._min_value, *values))
        self._max_value = float(max(self._max_value, *values))
        self._values.extend(float(v) for v in values)
