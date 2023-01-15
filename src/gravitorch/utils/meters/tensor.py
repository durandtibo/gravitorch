__all__ = ["ExtremaTensorMeter", "MeanTensorMeter", "TensorMeter", "TensorMeter2"]

from collections.abc import Iterable
from typing import Any, Optional, Union

import torch
from coola import objects_are_equal
from torch import Tensor

from gravitorch.distributed.ddp import MAX, MIN, SUM, sync_reduce
from gravitorch.utils.format import to_pretty_dict_str
from gravitorch.utils.meters.exceptions import EmptyMeterError
from gravitorch.utils.tensor import scalable_quantile
from gravitorch.utils.tensor.flatted import LazyFlattedTensor


class MeanTensorMeter:
    r"""Implements a meter to compute the mean value of ``torch.Tensor``s.

    The mean value is updated by keeping local variables ``total``
    and ``count``. ``count`` tracks the number of values, and
    ``total`` tracks the sum of the values.
    This meter has a constant space complexity.

    Args:
        count (int, optional): Specifies the initial count value.
            Default: ``0``
        total (float, optional): Specifies the initial total value.
            Default: ``0.0``

    Example usage:

    .. code-block:: python

        >>> import torch
        >>> from gravitorch.utils.meters import TensorMeter
        >>> meter = MeanTensorMeter()
        >>> meter.update(torch.arange(6))
        >>> meter.update(torch.tensor([4.0, 1.0]))
        >>> meter.mean()  # or meter.average()
        2.5
        >>> meter.sum()
        20.0
        >>> meter.count
        8
    """

    def __init__(self, count: int = 0, total: Union[int, float] = 0):
        self._count = int(count)
        self._total = total

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(count={self._count:,}, total={self._total})"

    @property
    def count(self) -> int:
        r"""int: The number of predictions in the meter."""
        return self._count

    @property
    def total(self) -> Union[int, float]:
        r"""int or float: The total sum value in the meter."""
        return self._total

    def reset(self) -> None:
        r"""Reset the meter."""
        self._count = 0
        self._total = 0

    def update(self, tensor: Tensor) -> None:
        r"""Updates the meter given a new tensor.

        Args:
            tensor (``torch.Tensor``): Specifies the new tensor to add
                to the meter.
        """
        self._total += tensor.sum().item()
        self._count += tensor.numel()

    def average(self) -> float:
        r"""Computes the average value.

        Returns:
            float: The average value.

        Raises:
            ``EmptyMeterError`` if the meter is empty.
        """
        return self.mean()

    def mean(self) -> float:
        r"""Gets the mean value.

        Returns:
            float: The mean value.

        Raises:
            ``EmptyMeterError`` if the meter is empty.
        """
        if not self._count:
            raise EmptyMeterError("The tensor meter is empty")
        return float(self._total) / float(self._count)

    def sum(self) -> Union[int, float]:
        r"""Gets the sum of all the values.

        Returns:
            int or float: The sum of all the values.

        Raises:
            ``EmptyMeterError`` if the meter is empty.
        """
        if not self._count:
            raise EmptyMeterError("The meter is empty")
        return self._total

    def all_reduce(self) -> "MeanTensorMeter":
        r"""Reduces the meter values across all machines in such a way that all
        get the final result.

        The sum value is reduced by summing all the sum values (1 sum
        value per distributed process). The count value is reduced by
        summing all the count values (1 count value per distributed
        process).

        In a non-distributed setting, this method returns a copy of
        the current meter.

        Returns:
            ``MeanTensorMeter``: The reduced meter.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from gravitorch.utils.meters import MeanTensorMeter
            >>> meter = MeanTensorMeter()
            >>> meter.update(torch.arange(6))
            >>> reduced_meter = meter.all_reduce()
        """
        return MeanTensorMeter(
            count=sync_reduce(self._count, SUM), total=sync_reduce(self._total, SUM)
        )

    def clone(self) -> "MeanTensorMeter":
        r"""Creates a copy of the current meter.

        Returns:
            ``MeanTensorMeter``: A copy of the current meter.
        """
        return MeanTensorMeter(count=self._count, total=self._total)

    def equal(self, other) -> bool:
        r"""Indicates if two meters are equal or not.

        Args:
            other: Specifies the value to compare.

        Returns:
            bool: ``True`` if the meters are equal,
                ``False`` otherwise.
        """
        if not isinstance(other, MeanTensorMeter):
            return False
        return self.state_dict() == other.state_dict()

    def merge(self, meters: Iterable["MeanTensorMeter"]) -> "MeanTensorMeter":
        r"""Merges several meters with the current meter and returns a new
        meter.

        Args:
            meters (iterable): Specifies the meters to merge to the
                current meter.

        Returns:
            ``MeanTensorMeter``: The merged meter.
        """
        count, total = self.count, self.total
        for meter in meters:
            count += meter.count
            total += meter.total
        return MeanTensorMeter(total=total, count=count)

    def merge_(self, meters: Iterable["MeanTensorMeter"]) -> None:
        r"""Merges several meters into the current meter.

        In-place version of ``merge``.

        Args:
            meters (iterable): Specifies the meters to merge to the
                current meter.
        """
        for meter in meters:
            self._count += meter.count
            self._total += meter.total

    def load_state_dict(self, state_dict: dict) -> None:
        r"""Loads a state to the history tracker.

        Args:
            state_dict (dict): Specifies a dictionary containing state
                keys with values.
        """
        self._count = state_dict["count"]
        self._total = state_dict["total"]

    def state_dict(self) -> dict[str, Union[int, float]]:
        r"""Returns a dictionary containing state values.

        Returns:
            dict: The state values in a dict.
        """
        return {"count": self._count, "total": self._total}


class ExtremaTensorMeter:
    r"""Implements a meter to compute the minimum and maximum values of
    ``torch.Tensor``s.

    The mean value is updated by keeping local variables ``min_value``
    and ``max_value``. ``min_value`` tracks the minimum value, and
    ``max_value`` tracks the maximum value.
    This meter has a constant space complexity.

    Args:
        count (int, optional): Specifies the initial count value.
            Default: ``0``
        min_value (int, optional): Specifies the initial minimum
            value. Default: ``inf``
        max_value (int, optional): Specifies the initial maximum
            value. Default: ``-inf``

    Example usage:

    .. code-block:: python

        >>> import torch
        >>> from gravitorch.utils.meters import ExtremaTensorMeter
        >>> meter = ExtremaTensorMeter()
        >>> meter.update(torch.arange(6))
        >>> meter.update(torch.tensor([4.0, 1.0]))
        >>> meter.max()
        5
        >>> meter.min()
        0
    """

    def __init__(
        self, count: int = 0, min_value: float = float("inf"), max_value: float = float("-inf")
    ):
        self._count = int(count)
        self._min_value = float(min_value)
        self._max_value = float(max_value)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(count={self._count:,}, "
            f"min_value={self._min_value}, max_value={self._max_value})"
        )

    @property
    def count(self) -> int:
        r"""int: The number of predictions in the meter."""
        return self._count

    def reset(self) -> None:
        r"""Reset the meter."""
        self._count = 0
        self._max_value = float("-inf")
        self._min_value = float("inf")

    def update(self, tensor: Tensor) -> None:
        r"""Updates the meter given a new tensor.

        Args:
            tensor (``torch.Tensor``): Specifies the new tensor to add
                to the meter.
        """
        min_value, max_value = torch.aminmax(tensor)
        self._max_value = max(self._max_value, max_value.item())
        self._min_value = min(self._min_value, min_value.item())
        self._count += tensor.numel()

    def max(self) -> float:
        r"""Gets the max value.

        Returns:
            float: The max value.

        Raises:
            ``EmptyMeterError`` if the meter is empty.
        """
        if not self._count:
            raise EmptyMeterError("The meter is empty")
        return float(self._max_value)

    def min(self) -> float:
        r"""Gets the min value.

        Returns:
            float: The min value.

        Raises:
            ``EmptyMeterError`` if the meter is empty.
        """
        if not self._count:
            raise EmptyMeterError("The meter is empty")
        return float(self._min_value)

    def all_reduce(self) -> "ExtremaTensorMeter":
        r"""Reduces the meter values across all machines in such a way that all
        get the final result.

        The maximum value is reduced by computing the maximum between
        the maximum values (1 maximum value per distributed process).
        The minimum value is reduced by computing the minimum between
        the minimum values (1 minimum value per distributed process).

        Returns:
            ``TensorMeter``: The reduced meter.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from gravitorch.utils.meters import ExtremaTensorMeter
            >>> meter = ExtremaTensorMeter()
            >>> meter.update(torch.arange(6))
            >>> reduced_meter = meter.all_reduce()
        """
        return ExtremaTensorMeter(
            count=sync_reduce(self._count, SUM),
            min_value=sync_reduce(self._min_value, MIN),
            max_value=sync_reduce(self._max_value, MAX),
        )

    def clone(self) -> "ExtremaTensorMeter":
        r"""Creates a copy of the current meter.

        Returns:
            ``TensorMeter``: A copy of the current meter.
        """
        return ExtremaTensorMeter(
            count=self._count, min_value=self._min_value, max_value=self._max_value
        )

    def equal(self, other) -> bool:
        r"""Indicates if two meters are equal or not.

        Args:
            other: Specifies the value to compare.

        Returns:
            bool: ``True`` if the meters are equal,
                ``False`` otherwise.
        """
        if not isinstance(other, ExtremaTensorMeter):
            return False
        return self.state_dict() == other.state_dict()

    def merge(self, meters: Iterable["ExtremaTensorMeter"]) -> "ExtremaTensorMeter":
        r"""Merges several meters with the current meter and returns a new
        meter.

        Args:
            meters (iterable): Specifies the meters to merge to the
                current meter.

        Returns:
            ``ExtremaTensorMeter``: The merged meter.
        """
        count, min_value, max_value = self._count, self._min_value, self._max_value
        for meter in meters:
            count += meter.count
            min_value = min(min_value, meter._min_value)
            max_value = max(max_value, meter._max_value)
        return ExtremaTensorMeter(count=count, min_value=min_value, max_value=max_value)

    def merge_(self, meters: Iterable["ExtremaTensorMeter"]) -> None:
        r"""Merges several meters into the current meter.

        In-place version of ``merge``.

        Args:
            meters (iterable): Specifies the meters to merge to the
                current meter.
        """
        for meter in meters:
            self._count += meter.count
            self._min_value = min(self._min_value, meter._min_value)
            self._max_value = max(self._max_value, meter._max_value)

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        r"""Loads a state to the history tracker.

        Args:
            state_dict (dict): Specifies a dictionary containing state
                keys with values.
        """
        self._count = state_dict["count"]
        self._min_value = state_dict["min_value"]
        self._max_value = state_dict["max_value"]

    def state_dict(self) -> dict[str, Any]:
        r"""Returns a dictionary containing state values.

        Returns:
            dict: The state values in a dict.
        """
        return {"count": self._count, "max_value": self._max_value, "min_value": self._min_value}


class TensorMeter:
    r"""Defines a class to compute and store the sum, average, maximum and
    minimum values of ``torch.Tensor``s.

    This meter has a constant space complexity.

    Args:
        count (int, optional): Specifies the initial count value.
            Default: ``0``
        total (float, optional): Specifies the initial sum value.
            Default: ``0.0``
        min_value (int, optional): Specifies the initial minimum
            value. Default: ``inf``
        max_value (int, optional): Specifies the initial maximum
            value. Default: ``-inf``

    Example usage:

    .. code-block:: python

        >>> import torch
        >>> from gravitorch.utils.meters import TensorMeter
        >>> meter = TensorMeter()
        >>> meter.update(torch.arange(6))
        >>> meter.update(torch.tensor([4.0, 1.0]))
        >>> meter.average()  # or meter.mean()
        2.5
        >>> meter.max()
        5
        >>> meter.min()
        0
        >>> meter.sum()
        20.0
        >>> meter.count
        8
    """

    def __init__(
        self,
        count: int = 0,
        total: float = 0.0,
        min_value: float = float("inf"),
        max_value: float = float("-inf"),
    ):
        self._count = int(count)
        self._total = float(total)
        self._min_value = float(min_value)
        self._max_value = float(max_value)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(count={self._count:,}, total={self._total}, "
            f"min_value={self._min_value}, max_value={self._max_value})"
        )

    def __str__(self) -> str:
        count = self.count
        stats = to_pretty_dict_str(
            {
                "count": f"{count:,}",
                "sum": self.sum() if count else "N/A (empty)",
                "average": self.average() if count else "N/A (empty)",
                "min": self.min() if count else "N/A (empty)",
                "max": self.max() if count else "N/A (empty)",
            },
            indent=2,
        )
        return f"{self.__class__.__qualname__}\n{stats}"

    @property
    def count(self) -> int:
        r"""int: The number of predictions in the meter."""
        return self._count

    @property
    def total(self) -> Union[int, float]:
        r"""int or float: The total sum value in the meter."""
        return self._total

    def all_reduce(self) -> "TensorMeter":
        r"""Reduces the meter values across all machines in such a way that all
        get the final result.

        The sum value is reduced by summing all the sum values (1 sum
        value per distributed process). The count value is reduced by
        summing all the count values (1 count value per distributed
        process). The maximum value is reduced by computing the
        maximum between the maximum values (1 maximum value per
        distributed process). The minimum value is reduced by
        computing the minimum between the minimum values (1 minimum
        value per distributed process).

        Returns:
            ``TensorMeter``: The reduced meter.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from gravitorch.utils.meters import TensorMeter
            >>> meter = TensorMeter()
            >>> meter.update(torch.arange(6))
            >>> reduced_meter = meter.all_reduce()
        """
        return TensorMeter(
            count=sync_reduce(self._count, SUM),
            total=sync_reduce(self._total, SUM),
            min_value=sync_reduce(self._min_value, MIN),
            max_value=sync_reduce(self._max_value, MAX),
        )

    def clone(self) -> "TensorMeter":
        r"""Creates a copy of the current meter.

        Returns:
            ``TensorMeter``: A copy of the current meter.
        """
        return TensorMeter(
            count=self._count,
            total=self._total,
            min_value=self._min_value,
            max_value=self._max_value,
        )

    def equal(self, other) -> bool:
        r"""Indicates if two meters are equal or not.

        Args:
            other: Specifies the value to compare.

        Returns:
            bool: ``True`` if the meters are equal,
                ``False`` otherwise.
        """
        if not isinstance(other, TensorMeter):
            return False
        return self.state_dict() == other.state_dict()

    def merge(self, meters: Iterable["TensorMeter"]) -> "TensorMeter":
        r"""Merges several meters with the current meter and returns a new
        meter.

        Args:
            meters (iterable): Specifies the meters to merge to the
                current meter.

        Returns:
            ``TensorMeter``: The merged meter.
        """
        count, total = self._count, self._total
        min_value, max_value = self._min_value, self._max_value
        for meter in meters:
            count += meter.count
            total += meter.total
            min_value = min(min_value, meter._min_value)
            max_value = max(max_value, meter._max_value)
        return TensorMeter(total=total, count=count, min_value=min_value, max_value=max_value)

    def merge_(self, meters: Iterable["TensorMeter"]) -> None:
        r"""Merges several meters into the current meter.

        In-place version of ``merge``.

        Args:
            meters (iterable): Specifies the meters to merge to the
                current meter.
        """
        for meter in meters:
            self._count += meter.count
            self._total += meter.total
            self._min_value = min(self._min_value, meter._min_value)
            self._max_value = max(self._max_value, meter._max_value)

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        r"""Loads a state to the history tracker.

        Args:
            state_dict (dict): Specifies a dictionary containing state
                keys with values.
        """
        self._count = state_dict["count"]
        self._max_value = state_dict["max_value"]
        self._min_value = state_dict["min_value"]
        self._total = state_dict["total"]

    def state_dict(self) -> dict[str, Any]:
        r"""Returns a dictionary containing state values.

        Returns:
            dict: The state values in a dict.
        """
        return {
            "count": self._count,
            "max_value": self._max_value,
            "min_value": self._min_value,
            "total": self._total,
        }

    def reset(self) -> None:
        r"""Reset the meter."""
        self._count = 0
        self._max_value = float("-inf")
        self._min_value = float("inf")
        self._total = 0.0

    def update(self, tensor: Tensor) -> None:
        r"""Updates the meter given a new tensor.

        Args:
            tensor (``torch.Tensor``): Specifies the new tensor to add
                to the meter.
        """
        min_value, max_value = torch.aminmax(tensor)
        self._max_value = max(self._max_value, max_value.item())
        self._min_value = min(self._min_value, min_value.item())
        self._total += tensor.sum().item()
        self._count += tensor.numel()

    def average(self) -> float:
        r"""Computes the average value.

        Returns:
            float: The average value.

        Raises:
            ``EmptyMeterError`` if the meter is empty.
        """
        if not self._count:
            raise EmptyMeterError("The tensor meter is empty")
        return self._total / float(self._count)

    def max(self) -> float:
        r"""Gets the max value.

        Returns:
            float: The max value.

        Raises:
            ``EmptyMeterError`` if the meter is empty.
        """
        if not self._count:
            raise EmptyMeterError("The meter is empty")
        return float(self._max_value)

    def mean(self) -> float:
        r"""Gets the mean value.

        Returns:
            float: The mean value.

        Raises:
            ``EmptyMeterError`` if the meter is empty.
        """
        if not self._count:
            raise EmptyMeterError("The tensor meter is empty")
        return self._total / float(self._count)

    def min(self) -> float:
        r"""Gets the min value.

        Returns:
            float: The min value.

        Raises:
            ``EmptyMeterError`` if the meter is empty.
        """
        if not self._count:
            raise EmptyMeterError("The meter is empty")
        return float(self._min_value)

    def sum(self) -> float:
        r"""Gets the sum of all the values.

        Returns:
            float: The sum of all the values.

        Raises:
            ``EmptyMeterError`` if the meter is empty.
        """
        if not self._count:
            raise EmptyMeterError("The meter is empty")
        return float(self._total)


class TensorMeter2:
    r"""Implements a meter to compute some stats on ``torch.Tensor``s.

    This meter has a linear space complexity.

    Args:
        values (``torch.Tensor`` or ``None``, optional): Specifies the
            initial values. The tensor is flattened if necessary.
            ``None`` means no initial values. Default: ``None``

    Example usage:

    .. code-block:: python

        >>> import torch
        >>> from gravitorch.utils.meters import TensorMeter2
        >>> meter = TensorMeter2()
        >>> meter.update(torch.arange(6))
        >>> meter.update(torch.tensor([4.0, 1.0]))
        >>> meter.count
        8
        >>> meter.average()  # or meter.mean()
        2.5
        >>> meter.max()
        5.0
        >>> meter.min()
        0.0
        >>> meter.sum()
        20.0
        >>> meter.median()
        2.0
        >>> meter.quantile(torch.tensor([0.1, 0.5]))
        tensor([0.7000, 2.5000])
        >>> meter.std()
        1.7728105783462524
    """

    def __init__(self, values: Optional[Tensor] = None):
        self._values = LazyFlattedTensor(values)
        self._count = self._values.numel()

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(count={self._count:,})"

    @property
    def count(self) -> int:
        r"""int: The number of predictions in the meter."""
        return self._count

    def reset(self) -> None:
        r"""Reset the meter."""
        self._count = 0
        self._values.clear()

    def update(self, tensor: Tensor) -> None:
        r"""Updates the meter given a new tensor.

        Args:
            tensor (``torch.Tensor``): Specifies the new tensor to add
                to the meter.
        """
        self._values.update(tensor.detach())
        self._count += tensor.numel()

    def average(self) -> float:
        r"""Computes the average value.

        Returns:
            float: The average value.

        Raises:
            ``EmptyMeterError`` if the meter is empty.
        """
        return self.mean()

    def max(self) -> Union[int, float]:
        r"""Gets the max value.

        Returns:
            int or float: The max value.

        Raises:
            ``EmptyMeterError`` if the meter is empty.
        """
        if not self._count:
            raise EmptyMeterError("The meter is empty")
        return self._values.values().max().item()

    def mean(self) -> float:
        r"""Gets the mean value.

        Returns:
            float: The mean value.

        Raises:
            ``EmptyMeterError`` if the meter is empty.
        """
        if not self._count:
            raise EmptyMeterError("The tensor meter is empty")
        return self._values.values().float().mean().item()

    def median(self) -> float:
        r"""Gets the median value.

        Returns:
            float: The median value.

        Raises:
            ``EmptyMeterError`` if the meter is empty.
        """
        if not self._count:
            raise EmptyMeterError("The meter is empty")
        return self._values.values().median().item()

    def min(self) -> Union[int, float]:
        r"""Gets the min value.

        Returns:
            int or float: The min value.

        Raises:
            ``EmptyMeterError`` if the meter is empty.
        """
        if not self._count:
            raise EmptyMeterError("The meter is empty")
        return self._values.values().min().item()

    def quantile(self, q: Tensor, method: str = "linear") -> Tensor:
        r"""Computes the ``q``-th quantiles.

        Args:
            q (``torch.Tensor`` of type float and shape
                ``(num_q_values,)``): Specifies the ``q``-values in
                the range ``[0, 1]``.
            method (str, optional): Specifies the interpolation
                method to use when the desired quantile lies between
                two data points. Can be ``'linear'``, ``'lower'``,
                ``'higher'``, ``'midpoint'`` and ``'nearest'``.
                Default: ``'linear'``.

        Returns:
            ``torch.Tensor`` of shape  ``(num_q_values,)``: The
                ``q``-th quantiles.

        Raises:
            ``EmptyMeterError`` if the meter is empty.
        """
        if not self._count:
            raise EmptyMeterError("The meter is empty")
        return scalable_quantile(self._values.values().float(), q=q, method=method)

    def std(self) -> float:
        r"""Gets the standard deviation value.

        Returns:
            float: The standard deviation value.

        Raises:
            ``EmptyMeterError`` if the meter is empty .
        """
        if not self._count:
            raise EmptyMeterError("The meter is empty")
        return self._values.values().float().std(dim=0).item()

    def sum(self) -> Union[int, float]:
        r"""Gets the sum of all the values.

        Returns:
            float: The sum of all the values.

        Raises:
            ``EmptyMeterError`` if the meter is empty.
        """
        if not self._count:
            raise EmptyMeterError("The meter is empty")
        return self._values.values().sum().item()

    def all_reduce(self) -> "TensorMeter2":
        r"""Reduces the meter values across all machines in such a way that all
        get the final result.

        Returns:
            ``TensorMeter2``: The reduced meter.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from gravitorch.utils.meters import TensorMeter2
            >>> meter = TensorMeter2()
            >>> meter.update(torch.arange(6))
            >>> reduced_meter = meter.all_reduce()
        """
        return TensorMeter2(self._values.all_reduce().values())

    def clone(self) -> "TensorMeter2":
        r"""Creates a copy of the current meter.

        Returns:
            ``TensorMeter2``: A copy of the current meter.
        """
        return TensorMeter2(self._values.clone().values())

    def equal(self, other) -> bool:
        r"""Indicates if two meters are equal or not.

        Args:
            other: Specifies the value to compare.

        Returns:
            bool: ``True`` if the meters are equal,
                ``False`` otherwise.
        """
        if not isinstance(other, TensorMeter2):
            return False
        return objects_are_equal(self.state_dict(), other.state_dict())

    def merge(self, meters: Iterable["TensorMeter2"]) -> "TensorMeter2":
        r"""Merges several meters with the current meter and returns a new
        meter.

        Args:
            meters (iterable): Specifies the meters to merge to the
                current meter.

        Returns:
            ``TensorMeter2``: The merged meter.
        """
        values = self._values.clone()
        for meter in meters:
            values.update(meter._values.values())
        return TensorMeter2(values.values())

    def merge_(self, meters: Iterable["TensorMeter2"]) -> None:
        r"""Merges several meters into the current meter.

        In-place version of ``merge``.

        Args:
            meters (iterable): Specifies the meters to merge to the
                current meter.
        """
        for meter in meters:
            self._values.update(meter._values.values())

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        r"""Loads a state to the history tracker.

        Args:
            state_dict (dict): Specifies a dictionary containing state
                keys with values.
        """
        self._values = LazyFlattedTensor(state_dict["values"])

    def state_dict(self) -> dict[str, Tensor]:
        r"""Returns a dictionary containing state values.

        Returns:
            dict: The state values in a dict.
        """
        return {"values": self._values.values()}
