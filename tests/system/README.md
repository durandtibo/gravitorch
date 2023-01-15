# System tests

## Slurm distributed launcher

To launch 2 processes on a single node, you can use the following command:

```bash
python -m gravitorch.cli.run -cd=tests/system/conf -cn=distributed_launcher
```

To run on a specific node, you can use the following command

```bash
python -m gravitorch.cli.run -cd=tests/system/conf -cn=distributed_launcher distributed.launcher.sbatch_variable=['--nodelist\=NODE_NAME']
```

If there is no GPU available, you can add `distributed.launcher.num_gpus_per_proc=0` to your command
to indicate you do
not want to use GPUs.

To launch 4 processes (2 nodes and 2 processes per node), you can use the following command:

```bash
python -m gravitorch.cli.run -cd=tests/system/conf -cn=distributed_launcher distributed/launcher=slurm_2x2
```

The logs of the application processes should look like:

```textmate
hostname=XXXXXXXX
MASTER_ADDR=XXXXXXXX
MASTER_PORT=XXXX
WORLD_SIZE=4
RANK=0
LOCAL_RANK=0

is_gloo_available True
is_nccl_available True
[0/4] is_initialized True
[0/4] backend gloo
[0/4] get_world_size 4
[0/4] get_rank 0
[0/4] device_count: 8
[local_rank=0] current_device: cuda:0

hostname=XXXXXXXX
MASTER_ADDR=XXXXXXXX
MASTER_PORT=XXXX
WORLD_SIZE=4
RANK=1
LOCAL_RANK=1

is_gloo_available True
is_nccl_available True
[1/4] is_initialized True
[1/4] backend gloo
[1/4] get_world_size 4
[1/4] get_rank 1
[1/4] device_count: 8
[local_rank=1] current_device: cuda:1

hostname=XXXXXXXX
MASTER_ADDR=XXXXXXXX
MASTER_PORT=XXXX
WORLD_SIZE=4
RANK=2
LOCAL_RANK=0

is_gloo_available True
is_nccl_available True
[2/4] is_initialized True
[2/4] backend gloo
[2/4] get_world_size 4
[2/4] get_rank 2
[2/4] device_count: 8
[local_rank=0] current_device: cuda:0

hostname=XXXXXXXX
MASTER_ADDR=XXXXXXXX
MASTER_PORT=XXXX
WORLD_SIZE=4
RANK=3
LOCAL_RANK=1

is_gloo_available True
is_nccl_available True
[3/4] is_initialized True
[3/4] backend gloo
[3/4] get_world_size 4
[3/4] get_rank 3
[3/4] device_count: 8
[local_rank=1] current_device: cuda:1
```

To verify that the application processes are correctly setup, you can verify the following
information:

- You should see 2 hostnames because there are 2 nodes. (`ulocpgpup100.fg.rbc.com`
  and `ulocpgpup100.fg.rbc.com` in this
  example)
- The world size should be `4` because there are 4 processes.
- Each process should have a unique rank: `0`, `1`, `2`, or `3`
- Each process should have a local rank: `0` or `1`
- The PyTorch context is initialized (`is_initialized`) and the distributed backend (`gloo` in this
  example).
- The device of each process is `cuda:0` or `cuda:1`.

To verify that the `srun` processes are executed in parallel, you can check the `sacct` logs.

```textmate
sacct --format=JobID,Start,End,Elapsed,NCPUS -j 453345
       JobID               Start                 End    Elapsed      NCPUS
------------ ------------------- ------------------- ---------- ----------
453345       2021-03-18T11:16:01             Unknown   00:02:38          4
453345.batch 2021-03-18T11:16:01             Unknown   00:02:38          2
453345.0     2021-03-18T11:16:40 2021-03-18T11:18:39   00:01:59          2
453345.1     2021-03-18T11:16:40 2021-03-18T11:18:37   00:01:57          2
```
