import datetime
from collections.abc import Generator

from ignite.distributed import Parallel
from pytest import fixture

from gravitorch.distributed.comm import Backend


@fixture(scope="session")
def parallel_gloo_2() -> Generator[Parallel, None, None]:
    with Parallel(
        backend=Backend.GLOO,
        nproc_per_node=2,
        nnodes=1,
        master_addr="localhost",
        master_port=29507,
        timeout=datetime.timedelta(seconds=60),
    ) as parallel:
        yield parallel


@fixture(scope="session")
def parallel_nccl_2() -> Generator[Parallel, None, None]:
    with Parallel(
        backend=Backend.NCCL,
        nproc_per_node=2,
        nnodes=1,
        master_addr="localhost",
        master_port=29508,
        timeout=datetime.timedelta(seconds=60),
    ) as parallel:
        yield parallel
