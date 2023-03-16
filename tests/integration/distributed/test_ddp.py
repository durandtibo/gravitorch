from collections.abc import Callable

import torch
from coola import objects_are_equal
from ignite.distributed import Parallel
from pytest import mark, raises

from gravitorch.distributed import comm as dist
from gravitorch.distributed import ddp
from gravitorch.distributed.ddp import all_gather_tensor_varshape
from gravitorch.testing import (
    distributed_available,
    gloo_available,
    nccl_available,
    two_gpus_available,
)

###########################################
#     Tests for broadcast_object_list     #
###########################################


def check_broadcast_object_list(local_rank: int) -> None:
    r"""Checks the ``broadcast_object_list``.

    Args:
    ----
        local_rank (int)-> None: Specifies the local rank.
    """
    assert dist.get_world_size() == 2  # This test is valid only for 2 processes.
    device = dist.device()

    # List of integers
    object_list = [0, 1, 3] if local_rank == 0 else [10, 11, 12]
    ddp.broadcast_object_list(object_list)
    assert objects_are_equal(object_list, [0, 1, 3])

    # List of tensors
    object_list = (
        [torch.tensor([0, 1], device=device)]
        if local_rank == 0
        else [torch.tensor([2, 2], device=device)]
    )
    ddp.broadcast_object_list(object_list)
    assert objects_are_equal(object_list, [torch.tensor([0, 1], device=device)])

    # List of integer and tensors
    object_list = (
        [2, torch.tensor([0, 1], device=device)]
        if local_rank == 0
        else [10, torch.tensor([2, 2], device=device)]
    )
    ddp.broadcast_object_list(object_list)
    assert objects_are_equal(object_list, [2, torch.tensor([0, 1], device=device)])


@distributed_available
@gloo_available
def test_broadcast_object_list_gloo(parallel_gloo_2: Parallel) -> None:
    print("parallel_gloo_2", parallel_gloo_2)
    parallel_gloo_2.run(check_sync_reduce_inplace)


@two_gpus_available
@distributed_available
@nccl_available
def test_broadcast_object_list_nccl(parallel_nccl_2: Parallel) -> None:
    parallel_nccl_2.run(check_sync_reduce_inplace)


def check_sync_reduce_tensor_int(local_rank: int) -> None:
    r"""This function checks the sync_reduce for an integer tensor input.

    Args:
    ----
        local_rank (int): Specifies the local rank.
    """
    assert dist.get_world_size() == 2  # This test is valid only for 2 processes.
    device = dist.device()

    x_tensor = (
        torch.tensor([0, 1], device=device)
        if local_rank == 0
        else torch.tensor([2, 2], device=device)
    )
    assert ddp.sync_reduce(x_tensor, op=ddp.AVG).equal(
        torch.tensor([1, 1.5], device=device)
    )  # average
    assert ddp.sync_reduce(x_tensor, op=ddp.MAX).equal(torch.tensor([2, 2], device=device))  # max
    assert ddp.sync_reduce(x_tensor, op=ddp.MIN).equal(torch.tensor([0, 1], device=device))  # min
    assert ddp.sync_reduce(x_tensor, op=ddp.PRODUCT).equal(
        torch.tensor([0, 2], device=device)
    )  # product
    assert ddp.sync_reduce(x_tensor, op=ddp.SUM).equal(torch.tensor([2, 3], device=device))  # sum

    if dist.backend() != dist.Backend.NCCL:  # bitwise AND and OR are not supported by NCCL
        assert ddp.sync_reduce(x_tensor, op=ddp.BAND).equal(
            torch.tensor([0, 0], device=device)
        )  # bitwise AND
        assert ddp.sync_reduce(x_tensor, op=ddp.BOR).equal(
            torch.tensor([2, 3], device=device)
        )  # bitwise OR

    # Verify that the original value did not change
    if local_rank == 0:
        assert x_tensor.equal(torch.tensor([0, 1], device=device))
    else:
        assert x_tensor.equal(torch.tensor([2, 2], device=device))


def check_sync_reduce_tensor_float(local_rank: int) -> None:
    r"""This function checks the sync_reduce for a float tensor input.

    Args:
    ----
        local_rank (int): Specifies the local rank.
    """
    assert dist.get_world_size() == 2  # This test is valid only for 2 processes.
    device = dist.device()

    x_tensor = (
        torch.tensor([0.0, 1.0], device=device)
        if local_rank == 0
        else torch.tensor([2.0, 2.0], device=device)
    )
    assert ddp.sync_reduce(x_tensor, op=ddp.AVG).equal(
        torch.tensor([1, 1.5], device=device)
    )  # average
    assert ddp.sync_reduce(x_tensor, op=ddp.MAX).equal(
        torch.tensor([2.0, 2.0], device=device)
    )  # max
    assert ddp.sync_reduce(x_tensor, op=ddp.MIN).equal(
        torch.tensor([0.0, 1.0], device=device)
    )  # min
    assert ddp.sync_reduce(x_tensor, op=ddp.PRODUCT).equal(
        torch.tensor([0.0, 2.0], device=device)
    )  # product
    assert ddp.sync_reduce(x_tensor, op=ddp.SUM).equal(
        torch.tensor([2.0, 3.0], device=device)
    )  # sum

    with raises(RuntimeError):
        ddp.sync_reduce(x_tensor, op=ddp.BAND)  # bitwise AND is not valid for float number
    with raises(RuntimeError):
        ddp.sync_reduce(x_tensor, op=ddp.BOR)  # bitwise OR is not valid for float number

    # Verify that the original value did not change
    if local_rank == 0:
        assert x_tensor.equal(torch.tensor([0.0, 1.0], device=device))
    else:
        assert x_tensor.equal(torch.tensor([2.0, 2.0], device=device))


def check_sync_reduce_int(local_rank: int) -> None:
    r"""This function checks the sync_reduce for a python integer input.

    Args:
    ----
        local_rank (int): Specifies the local rank.
    """
    assert dist.get_world_size() == 2  # This test is valid only for 2 processes.

    x_int = 2 if local_rank == 0 else 5
    assert ddp.sync_reduce(x_int, op=ddp.AVG) == 3.5  # average
    assert ddp.sync_reduce(x_int, op=ddp.MAX) == 5  # max
    assert ddp.sync_reduce(x_int, op=ddp.MIN) == 2  # min
    assert ddp.sync_reduce(x_int, op=ddp.PRODUCT) == 10  # product
    assert ddp.sync_reduce(x_int, op=ddp.SUM) == 7  # sum

    if dist.backend() != dist.Backend.NCCL:  # bitwise AND and OR are not supported by NCCL
        assert ddp.sync_reduce(x_int, op=ddp.BAND) == 0  # bitwise AND
        assert ddp.sync_reduce(x_int, op=ddp.BOR) == 7  # bitwise OR

    # Verify that the original value did not change
    if local_rank == 0:
        assert x_int == 2
    else:
        assert x_int == 5


def sync_reduce_float(local_rank: int) -> None:
    r"""This function checks the sync_reduce for a python integer float.

    It also checks that the operations are not done in-place i.e. the original value did not change.

    Args:
    ----
        local_rank (int): Specifies the local rank.
    """
    assert dist.get_world_size() == 2  # This test is valid only for 2 processes.

    x_float = 1.0 if local_rank == 0 else 3.5
    assert ddp.sync_reduce(x_float, op=ddp.AVG) == 2.25  # average
    assert ddp.sync_reduce(x_float, op=ddp.MAX) == 3.5  # max
    assert ddp.sync_reduce(x_float, op=ddp.MIN) == 1.0  # min
    assert ddp.sync_reduce(x_float, op=ddp.PRODUCT) == 3.5  # product
    assert ddp.sync_reduce(x_float, op=ddp.SUM) == 4.5  # sum

    with raises(RuntimeError):
        ddp.sync_reduce(x_float, op=ddp.BAND)  # bitwise AND is not valid for float number
    with raises(RuntimeError):
        ddp.sync_reduce(x_float, op=ddp.BOR)  # bitwise OR is not valid for float number

    # Verify that the original value did not change
    if local_rank == 0:
        assert x_float == 1.0
    else:
        assert x_float == 3.5


def check_sync_reduce_inplace(local_rank: int) -> None:
    assert dist.get_world_size() == 2
    device = dist.device()

    x_tensor = (
        torch.tensor([0.0, 1.0], device=device)
        if local_rank == 0
        else torch.tensor([2.0, 2.0], device=device)
    )
    assert ddp.sync_reduce_(x_tensor, op=ddp.AVG).equal(
        torch.tensor([1, 1.5], device=device)
    )  # average
    assert x_tensor.equal(torch.tensor([1, 1.5], device=device))

    x_tensor = (
        torch.tensor([0, 1], device=device)
        if local_rank == 0
        else torch.tensor([2, 2], device=device)
    )
    assert ddp.sync_reduce_(x_tensor, op=ddp.MAX).equal(torch.tensor([2, 2], device=device))  # max
    assert x_tensor.equal(torch.tensor([2, 2], device=device))

    x_tensor = (
        torch.tensor([0, 1], device=device)
        if local_rank == 0
        else torch.tensor([2, 2], device=device)
    )
    assert ddp.sync_reduce_(x_tensor, op=ddp.MIN).equal(torch.tensor([0, 1], device=device))  # min
    assert x_tensor.equal(torch.tensor([0, 1], device=device))

    x_tensor = (
        torch.tensor([0, 1], device=device)
        if local_rank == 0
        else torch.tensor([2, 2], device=device)
    )
    assert ddp.sync_reduce_(x_tensor, op=ddp.PRODUCT).equal(
        torch.tensor([0, 2], device=device)
    )  # product
    assert x_tensor.equal(torch.tensor([0, 2], device=device))

    x_tensor = (
        torch.tensor([0, 1], device=device)
        if local_rank == 0
        else torch.tensor([2, 2], device=device)
    )
    assert ddp.sync_reduce_(x_tensor, op=ddp.SUM).equal(torch.tensor([2, 3], device=device))  # sum
    assert x_tensor.equal(torch.tensor([2, 3], device=device))

    if dist.backend() != dist.Backend.NCCL:  # bitwise AND and OR are not supported by NCCL
        x_tensor = (
            torch.tensor([0, 1], device=device)
            if local_rank == 0
            else torch.tensor([2, 2], device=device)
        )
        assert ddp.sync_reduce_(x_tensor, op=ddp.BAND).equal(
            torch.tensor([0, 0], device=device)
        )  # bitwise AND
        assert x_tensor.equal(torch.tensor([0, 0], device=device))

        x_tensor = (
            torch.tensor([0, 1], device=device)
            if local_rank == 0
            else torch.tensor([2, 2], device=device)
        )
        assert ddp.sync_reduce_(x_tensor, op=ddp.BOR).equal(
            torch.tensor([2, 3], device=device)
        )  # bitwise OR
        assert x_tensor.equal(torch.tensor([2, 3], device=device))

    with raises(TypeError):
        ddp.sync_reduce_(1.0, op=ddp.SUM)  # Does not support float

    with raises(TypeError):
        ddp.sync_reduce_(2, op=ddp.SUM)  # Does not support integer


@mark.parametrize(
    "func",
    [
        check_sync_reduce_tensor_int,
        check_sync_reduce_tensor_float,
        check_sync_reduce_int,
        sync_reduce_float,
    ],
)
@distributed_available
@gloo_available
def test_sync_reduce_gloo(parallel_gloo_2: Parallel, func: Callable) -> None:
    print("parallel_gloo_2", parallel_gloo_2)
    parallel_gloo_2.run(func)


@distributed_available
@gloo_available
def test_sync_reduce_inplace_gloo(parallel_gloo_2: Parallel) -> None:
    print("parallel_gloo_2", parallel_gloo_2)
    parallel_gloo_2.run(check_sync_reduce_inplace)


@mark.parametrize(
    "func",
    [
        check_sync_reduce_tensor_int,
        check_sync_reduce_tensor_float,
        check_sync_reduce_int,
        sync_reduce_float,
    ],
)
@two_gpus_available
@distributed_available
@nccl_available
def test_sync_reduce_nccl(parallel_nccl_2: Parallel, func: Callable) -> None:
    parallel_nccl_2.run(func)


@two_gpus_available
@distributed_available
@nccl_available
def test_sync_reduce_inplace_nccl(parallel_nccl_2: Parallel) -> None:
    parallel_nccl_2.run(check_sync_reduce_inplace)


################################################
#     Tests for all_gather_tensor_varshape     #
################################################


def check_all_gather_tensor_varshape(local_rank: int) -> None:
    assert dist.get_world_size() == 2
    device = dist.device()

    # 1-d tensor
    assert objects_are_equal(
        all_gather_tensor_varshape(
            torch.tensor([0.0, 1.0]) if local_rank == 0 else torch.tensor([2.0, 3.0, 4.0])
        ),
        [
            torch.tensor([0.0, 1.0], dtype=torch.float, device=device),
            torch.tensor([2.0, 3.0, 4.0], dtype=torch.float, device=device),
        ],
    )

    # 2-d tensor
    assert objects_are_equal(
        all_gather_tensor_varshape(
            torch.tensor([[0, 1, 2], [3, 4, 5]]) if local_rank == 0 else torch.tensor([[1], [0]])
        ),
        [
            torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.long, device=device),
            torch.tensor([[1], [0]], dtype=torch.long, device=device),
        ],
    )

    # 3-d tensor
    assert objects_are_equal(
        all_gather_tensor_varshape(torch.ones(2, 3, 4) if local_rank == 0 else torch.ones(4, 3, 1)),
        [torch.ones(2, 3, 4, device=device), torch.ones(4, 3, 1, device=device)],
    )


@distributed_available
@gloo_available
def test_all_gather_tensor_varshape_gloo(parallel_gloo_2: Parallel) -> None:
    print("parallel_gloo_2", parallel_gloo_2)
    parallel_gloo_2.run(check_all_gather_tensor_varshape)


@two_gpus_available
@distributed_available
@nccl_available
def test_all_gather_tensor_varshape_nccl(parallel_nccl_2: Parallel) -> None:
    parallel_nccl_2.run(check_all_gather_tensor_varshape)
