__all__ = [
    # Nvidia
    "CUDA_VISIBLE_DEVICES",
    # PyTorch
    "GROUP_RANK",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "RANK",
    "ROLE_RANK",
    "ROLE_WORLD_SIZE",
    "TORCHELASTIC_MAX_RESTARTS",
    "TORCHELASTIC_RESTART_COUNT",
    "TORCHELASTIC_RUN_ID",
    "TORCH_DISTRIBUTED_ENV_VARS",
    "WORLD_SIZE",
    # SLURM
    "SLURM_DISTRIBUTED_ENV_VARS",
    "SLURM_JOB_ID",
    "SLURM_JOB_NODELIST",
    "SLURM_LOCALID",
    "SLURM_NTASKS",
    "SLURM_PROCID",
]

CUDA_VISIBLE_DEVICES = "CUDA_VISIBLE_DEVICES"

# The following constants defines the name of the environment variables
# used in PyTorch so it is not possible to change them. In particular,
# these constants are used by ``torch.distributed.run``
# https://github.com/pytorch/pytorch/blob/master/torch/distributed/run.py to launch and initialize the PyTorch
# distributed backend.
LOCAL_RANK = "LOCAL_RANK"  # The local rank.
RANK = "RANK"  # The global rank.
# The rank of the worker group. A number between 0 and ``max_nnodes``.
# When running a single worker group per node, this is the rank of the node.
GROUP_RANK = "GROUP_RANK"
# The rank of the worker across all the workers that have the same role.
# The role of the worker is specified in the ``WorkerSpec``.
ROLE_RANK = "ROLE_RANK"
LOCAL_WORLD_SIZE = (
    "LOCAL_WORLD_SIZE"  # The local world size (e.g. number of workers running locally).
)
WORLD_SIZE = "WORLD_SIZE"  # The world size (total number of workers in the job).
# The total number of workers that was launched with the same role specified in ``WorkerSpec``.
ROLE_WORLD_SIZE = "ROLE_WORLD_SIZE"
# The FQDN of the host that is running worker with rank 0; used to initialize
# the Torch Distributed backend.
MASTER_ADDR = "MASTER_ADDR"
MASTER_PORT = (
    "MASTER_PORT"  # The port on the ``MASTER_ADDR`` that can be used to host the C10d TCP store.
)
TORCHELASTIC_RESTART_COUNT = (
    "TORCHELASTIC_RESTART_COUNT"  # The number of worker group restarts so far.
)
TORCHELASTIC_MAX_RESTARTS = (
    "TORCHELASTIC_MAX_RESTARTS"  # The configured maximum number of restarts.
)
TORCHELASTIC_RUN_ID = (
    "TORCHELASTIC_RUN_ID"  # Equal to the rendezvous ``run_id`` (e.g. unique job id).
)
# The tuple of environment variable names used to initialize the PyTorch distributed backend.
TORCH_DISTRIBUTED_ENV_VARS = (
    GROUP_RANK,
    LOCAL_RANK,
    LOCAL_WORLD_SIZE,
    MASTER_ADDR,
    MASTER_PORT,
    RANK,
    ROLE_RANK,
    ROLE_WORLD_SIZE,
    TORCHELASTIC_MAX_RESTARTS,
    TORCHELASTIC_RESTART_COUNT,
    TORCHELASTIC_RUN_ID,
    WORLD_SIZE,
)

# These SLURM environment variables are used to initialize the native PyTorch
# distributed backends. They are used by PyTorch Ignite
# (see https://github.com/sdesrozis/why-ignite for more information on how these
# environment variables were chosen)
SLURM_JOB_ID = "SLURM_JOB_ID"
SLURM_JOB_NODELIST = "SLURM_JOB_NODELIST"
SLURM_LOCALID = "SLURM_LOCALID"
SLURM_NTASKS = "SLURM_NTASKS"
SLURM_PROCID = "SLURM_PROCID"
# The tuple of environment variable names used to initialize the
# PyTorch distributed backend in a SLURM environment.
SLURM_DISTRIBUTED_ENV_VARS = (
    SLURM_JOB_ID,
    SLURM_JOB_NODELIST,
    SLURM_LOCALID,
    SLURM_NTASKS,
    SLURM_PROCID,
)
