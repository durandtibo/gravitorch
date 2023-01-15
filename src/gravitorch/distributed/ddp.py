__all__ = ["all_gather_tensor_varshape", "broadcast_object_list", "sync_reduce", "sync_reduce_"]

from typing import Optional, Union, overload

import torch
from torch import Tensor
from torch import distributed as tdist

from gravitorch.distributed.comm import all_gather, all_reduce
from gravitorch.distributed.comm import device as dist_device
from gravitorch.distributed.comm import get_world_size, is_distributed

# The supported reduction operators
AVG = "AVG"
BAND = "AND"  # Bitwise AND (only for integer/long)
BOR = "OR"  # Bitwise OR (only for integer/long)
MAX = "MAX"
MIN = "MIN"
PRODUCT = "PRODUCT"
SUM = "SUM"


def broadcast_object_list(
    object_list: list, src: int = 0, device: Optional[torch.device] = None
) -> None:
    r"""Broadcasts picklable objects in ``object_list``.

    See ``torch.distributed.broadcast_object_list`` for more
    information.

    Args:
        object_list (List[Any]): List of input objects to broadcast.
            Each object must be picklable. Only objects on the
            ``src`` rank will be broadcast, but each rank must provide
            lists of equal sizes.
        src (int, optional): Specifies the source rank from which to
            broadcast ``object_list``. Default: ``0``
        device (``torch.device``, optional): If not ``None``, the
            objects are serialized and converted to tensors which are
            moved to the ``device`` before broadcasting.
            Default is ``None``.

    Example usage:

    .. code-block:: python

        >>> # Note: Process group initialization omitted on each rank.
        >>> import torch
        >>> from gravitorch import distributed as dist
        >>> x = [10 * dist.get_local_rank(), 10 * dist.get_local_rank() + 1]
        >>> print(x)
        [0, 1]    # Rank 0
        [10, 11]  # Rank 1
        >>> from gravitorch.distributed import ddp
        >>> ddp.broadcast_object_list(x)
        >>> print(x)
        [0, 1]    # Rank 0
        [0, 1]    # Rank 1
    """
    # There is nothing to broadcast in a non-distributed environment,
    if is_distributed():
        tdist.broadcast_object_list(object_list=object_list, src=src, device=device)


@overload
def sync_reduce(variable: Tensor, op: str) -> Tensor:
    r"""``sync_reduce`` for ``torch.Tensor``."""


@overload
def sync_reduce(variable: int, op: str) -> int:
    r"""``sync_reduce`` for int."""


@overload
def sync_reduce(variable: float, op: str) -> float:
    r"""``sync_reduce`` for float."""


def sync_reduce(variable: Union[Tensor, int, float], op: str) -> Union[Tensor, int, float]:
    r"""Synchronizes all the processes and then reduce the variable.

    This function is a no-operation function if the distributed mode
    is not activated. It returns the input. If the distributed mode
    is activated, this function does not change the input variable.
    If the input is a tensor, this function will create a copy of the
    tensor before to reduce it. After this function is executed,
    the input variable will contain the value before reduction.
    If you want to do an in-place operation, you can use
    ``sync_reduce_``.

    Args:
        variable (``torch.Tensor`` or int or float): Specifies the
            variable to reduce.
        op (str): Specifies the reduction operation.
            The available operations are: ``AVG``, ``AND``, ``OR``,
            ``MAX``, ``MIN``, ``PRODUCT`` and ``SUM``.

    Returns:
        ``torch.Tensor`` or int or float: The reduced variable.

    Example usage:

    .. code-block:: python

        >>> import torch
        >>> from gravitorch import distributed as dist
        >>> x = torch.ones(2, 3, device=dist.device())
        >>> from gravitorch.distributed import ddp
        >>> x_reduced = ddp.sync_reduce(x, op=ddp.SUM)
    """
    if is_distributed():
        divide_by_world_size = False
        if op == AVG:
            # Average is not a supported operation by PyTorch distributed.
            op = SUM
            divide_by_world_size = True
        if torch.is_tensor(variable):
            # Create a copy to not change the values of the input tensor.
            variable = variable.clone()
        variable = all_reduce(variable, op=op)
        if divide_by_world_size:
            variable = variable / get_world_size()
    return variable


def sync_reduce_(tensor: Tensor, op: str) -> Tensor:
    r"""In-place version of ``sync_reduce`` but it works only for a tensor.

    Args:
        tensor (``torch.Tensor``): Specifies the tensor to reduce.
        op (str): Specifies the reduction operation.
            The available operations are: ``AVG``, ``AND``, ``OR``,
            ``MAX``, ``MIN``, ``PRODUCT`` and ``SUM``.

    Returns:
        The reduced tensor which is also the input tensor.

    Raises:
        ``TypeError`` if the input is not a tensor.

    Example usage:

    .. code-block:: python

        >>> import torch
        >>> from gravitorch import distributed as dist
        >>> x = torch.ones(2, 3, device=dist.device())
        >>> from gravitorch.distributed import ddp
        >>> ddp.sync_reduce_(x, op=ddp.SUM)  # in-place reduction
    """
    if not torch.is_tensor(tensor):
        raise TypeError("The function sync_reduce_ only supports Tensor")

    if is_distributed():
        divide_by_world_size = False
        if op == AVG:
            # Average is not a supported operation by PyTorch distributed.
            op = SUM
            divide_by_world_size = True
        all_reduce(tensor, op=op)
        if divide_by_world_size:
            tensor.div_(get_world_size())
    return tensor


def all_gather_tensor_varshape(tensor: Tensor) -> list[Tensor]:
    r"""Implements an all gather operation for variable shape tensors.

    Note: the tensors can have variable shapes, but they have to have
    the same number of dimensions. The tensor should have at least one
    dimension.

    Args:
        tensor (``torch.Tensor``): Specifies the tensor to collect
            across participating processes.

    Returns:
        list: The list of collected tensors.

    Example usage:

    .. code-block:: python

        >>> import torch
        >>> x = torch.tensor([[0, 1, 2], [3, 4, 5]])  # process 0
        >>> x = torch.tensor([[1], [0]])  # process 1
        >>> from gravitorch.distributed import ddp
        >>> ddp.all_gather_tensor_varshape(x)
        [tensor([[0, 1, 2], [3, 4, 5]]), tensor([[1], [0]])]
    """
    if not is_distributed():
        return [tensor]

    shapes = all_gather(torch.tensor(tensor.shape).unsqueeze(dim=0))
    numels = shapes.prod(dim=1)
    tensor_padded = torch.zeros(numels.max().item(), dtype=tensor.dtype, device=dist_device())
    tensor_padded[: tensor.numel()] = tensor.flatten()
    tensors_padded = all_gather(tensor_padded.unsqueeze(dim=0))
    return [values[:n].view(*shape) for n, shape, values in zip(numels, shapes, tensors_padded)]
