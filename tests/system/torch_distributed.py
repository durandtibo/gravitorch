import os
import time
from datetime import timedelta

import torch
from torch import distributed as dist

from gravitorch.distributed import Backend
from gravitorch.distributed import _constants as ct
from gravitorch.distributed.url_helpers import get_hostname


def main():
    print(
        f"hostname={get_hostname()}\n"
        f"MASTER_ADDR={os.environ[ct.MASTER_ADDR]}\n"
        f"MASTER_PORT={os.environ[ct.MASTER_PORT]}\n"
        f"WORLD_SIZE={os.environ[ct.WORLD_SIZE]}\n"
        f"RANK={os.environ[ct.RANK]}\n"
        f"LOCAL_RANK={os.environ[ct.LOCAL_RANK]}\n"
        f"is_gloo_available={dist.is_gloo_available()}\n"
        f"is_nccl_available={dist.is_nccl_available()}\n"
    )
    backend = "nccl" if dist.is_nccl_available() and torch.cuda.is_available() else "gloo"

    world_size = int(os.environ.get(ct.WORLD_SIZE, -1))
    if dist.is_available() and world_size > 1:
        dist.init_process_group(backend, timeout=timedelta(minutes=2))
        local_rank = int(os.environ[ct.LOCAL_RANK])
        if dist.get_backend() == Backend.NCCL:
            dist.barrier(device_ids=[local_rank])
        else:
            dist.barrier()

        with torch.cuda.device(local_rank):  # Work only if CUDA is installed
            if dist.is_initialized():
                rank = dist.get_rank()
                world_size = dist.get_world_size()
                print(
                    f"[{rank}/{world_size}] is_initialized: {dist.is_initialized()}\n"
                    f"[{rank}/{world_size}] backend: {dist.get_backend()}\n"
                    f"[{rank}/{world_size}] get_world_size: {dist.get_world_size()}\n"
                    f"[{rank}/{world_size}] get_rank: {dist.get_rank()}\n"
                    f"[{rank}/{world_size}] device_count: {torch.cuda.device_count()}\n"
                )

            device = torch.device(f"cuda:{torch.cuda.current_device()}")
            print(f"[local_rank={local_rank}] current_device: {device}")
            dist.barrier()
            print(f"[{rank}/{world_size}] tensor: {torch.zeros(2, 3, device=device)}")

        time.sleep(10)  # wait 10 seconds

        print(f"[{rank}/{world_size}] destroy process group")
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
