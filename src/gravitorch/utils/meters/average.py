from __future__ import annotations

__all__ = ["AverageMeter"]

from collections.abc import Iterable
from typing import Any

from gravitorch.distributed.ddp import SUM, sync_reduce
from gravitorch.utils.format import str_pretty_dict
from gravitorch.utils.meters.exceptions import EmptyMeterError


class AverageMeter:
    r"""Defines a class to compute and store the average value of float
    number.

    Args:
    ----
        total (float, optional): Specifies the initial total value.
            Default: ``0.0``
        count (int, optional): Specifies the initial count value.
            Default: ``0``

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.utils.meters import AverageMeter
        >>> meter = AverageMeter()
        >>> for i in range(11):
        ...     meter.update(i)
        ...
        >>> meter.average()
        5.0
        >>> meter.sum()
        55.0
        >>> meter.count
        11
    """

    def __init__(self, total: float = 0.0, count: int = 0) -> None:
        self._total = float(total)
        self._count = int(count)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(count={self._count:,}, total={self._total})"

    def __str__(self) -> str:
        stats = str_pretty_dict(
            {
                "average": self.average() if self.count else "N/A (empty)",
                "count": self.count,
                "total": self.total,
            },
            indent=2,
        )
        return f"{self.__class__.__qualname__}\n{stats}"

    @property
    def count(self) -> int:
        r"""``int``: The number of examples in the meter since the last
        reset."""
        return self._count

    @property
    def total(self) -> float:
        r"""``float``: The total of the values added to the meter since
        the last reset."""
        return self._total

    def all_reduce(self) -> AverageMeter:
        r"""Reduces the meter values across all machines in such a way
        that all get the final result.

        The total value is reduced by summing all the sum values
        (1 total value per distributed process).
        The count value is reduced by summing all the count values
        (1 count value per distributed process).

        Returns
        -------
            ``AverageMeter``: The reduced meter.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import AverageMeter
            >>> meter = AverageMeter()
            >>> meter.update(6)
            >>> reduced_meter = meter.all_reduce()
        """
        return AverageMeter(
            total=sync_reduce(self._total, SUM),
            count=sync_reduce(self._count, SUM),
        )

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

            >>> from gravitorch.utils.meters import AverageMeter
            >>> meter = AverageMeter()
            >>> for i in range(11):
            ...     meter.update(i)
            ...
            >>> meter.average()
            5.0
        """
        if not self._count:
            raise EmptyMeterError("The meter is empty")
        return self._total / float(self._count)

    def clone(self) -> AverageMeter:
        r"""Creates a copy of the current meter.

        Returns
        -------
            ``AverageMeter``: A copy of the current meter.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from gravitorch.utils.meters import AverageMeter
            >>> meter = AverageMeter(total=55.0, count=11)
            >>> meter_cloned = meter.clone()
            >>> meter.update(1)
            >>> meter.sum()
            56.0
            >>> meter.count
            12
            >>> meter_cloned.sum()
            55.0
            >>> meter_cloned.count
            11
        """
        return AverageMeter(total=self.total, count=self.count)

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
            >>> from gravitorch.utils.meters import AverageMeter
            >>> meter1 = AverageMeter(total=55.0, count=11)
            >>> meter2 = AverageMeter(total=3.0, count=3)
            >>> meter1.equal(meter2)
            False
        """
        if not isinstance(other, AverageMeter):
            return False
        return self.state_dict() == other.state_dict()

    def merge(self, meters: Iterable[AverageMeter]) -> AverageMeter:
        r"""Merges several meters with the current meter and returns a
        new meter.

        Args:
        ----
            meters (iterable): Specifies the meters to merge to the
                current meter.

        Returns:
        -------
            ``AverageMeter``: The merged meter.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from gravitorch.utils.meters import AverageMeter
            >>> meter1 = AverageMeter(total=55.0, count=10)
            >>> meter2 = AverageMeter(total=3.0, count=3)
            >>> meter3 = meter1.merge([meter2])
            >>> meter3.count
            13
            >>> meter3.sum()
            58.0
        """
        count, total = self.count, self.total
        for meter in meters:
            count += meter.count
            total += meter.total
        return AverageMeter(total=total, count=count)

    def merge_(self, meters: Iterable[AverageMeter]) -> None:
        r"""Merges several meters into the current meter.

        In-place version of ``merge``.

        Args:
        ----
            meters (iterable): Specifies the meters to merge to the
                current meter.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from gravitorch.utils.meters import AverageMeter
            >>> meter1 = AverageMeter(total=55.0, count=10)
            >>> meter2 = AverageMeter(total=3.0, count=3)
            >>> meter1.merge_([meter2])
            >>> meter1.count
            13
            >>> meter1.sum()
            58.0
        """
        for meter in meters:
            self._count += meter.count
            self._total += meter.total

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        r"""Loads a state to the history tracker.

        Args:
        ----
            state_dict (dict): Specifies a dictionary containing state
                keys with values.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import AverageMeter
            >>> meter = AverageMeter()
            >>> meter.load_state_dict({"count": 11, "total": 55.0})
            >>> meter.count
            11
            >>> meter.sum()
            55.0
        """
        self._total = state_dict["total"]
        self._count = state_dict["count"]

    def reset(self) -> None:
        r"""Resets the meter.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import AverageMeter
            >>> meter = AverageMeter()
            >>> for i in range(11):
            ...     meter.update(i)
            ...
            >>> meter.reset()
            >>> meter.count
            0
        """
        self._total = 0.0
        self._count = 0

    def state_dict(self) -> dict[str, Any]:
        r"""Returns a dictionary containing state values.

        Returns
        -------
            dict: The state values in a dict.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import AverageMeter
            >>> meter = AverageMeter()
            >>> for i in range(11):
            ...     meter.update(i)
            ...
            >>> meter.state_dict()
            {'count': 11, 'total': 55.0}
        """
        return {"count": self._count, "total": self._total}

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

            >>> from gravitorch.utils.meters import AverageMeter
            >>> meter = AverageMeter()
            >>> for i in range(11):
            ...     meter.update(i)
            ...
            >>> meter.sum()
            55.0
        """
        if not self._count:
            raise EmptyMeterError("The meter is empty")
        return self._total

    def update(self, value: float, num_examples: int = 1) -> None:
        r"""Updates the meter given a new value and the number of
        examples.

        Args:
        ----
            value (float): Specifies the value to add to the meter.
            num_examples (int, optional): Specifies the number of
                examples. This argument is mainly used to deal with
                mini-batches of different sizes. Default: ``1``

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import AverageMeter
            >>> meter = AverageMeter()
            >>> for i in range(11):
            ...     meter.update(i)
            ...
            >>> meter.sum()
            55.0
        """
        self._total += float(value) * num_examples
        self._count += num_examples
