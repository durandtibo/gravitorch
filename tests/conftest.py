from ignite import distributed as idist
from pytest import fixture

from gravitorch.distributed.comm import Backend


@fixture
def parallel_gloo_2() -> idist.Parallel:
    return idist.Parallel(
        backend=Backend.GLOO,
        nproc_per_node=2,
        nnodes=1,
        master_addr="127.0.0.1",
        master_port=29507,
        daemon=True,
    )


@fixture
def parallel_nccl_2() -> idist.Parallel:
    return idist.Parallel(
        backend=Backend.NCCL,
        nproc_per_node=2,
        nnodes=1,
        master_addr="127.0.0.1",
        master_port=29508,
        daemon=True,
    )
