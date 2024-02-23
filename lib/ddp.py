import sys
import os
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import random
from collections import defaultdict

def _worker_fn(rank, world_size, main_fn, args_dict):
    # Setup
    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl', rank=rank,
        world_size=world_size)
    if rank != 0:
        sys.stdout = open('/dev/null', 'w')

    # Main function
    main_fn(**args_dict)

    # Cleanup
    dist.destroy_process_group()


def _torchrun_worker_fn(main_fn, args_dict):
    # Setup
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    print(f'Rank: {rank()}/{world_size()} (local rank {local_rank})')
    if local_rank != 0:
        sys.stdout = open('/dev/null', 'w')

    # Main function
    main_fn(**args_dict)

    # Cleanup
    dist.destroy_process_group()


def wrap_main(main_fn):
    """
    Usage: instead of calling main() directly, call wrap_main(main)().
    main should take only kwargs.
    """
    world_size = torch.cuda.device_count()
    def main(**args):
        if 'RANK' in os.environ:
            mp.set_start_method('spawn')
            _torchrun_worker_fn(main_fn, args)
        else:
            os.environ['PYTHONUNBUFFERED'] = '1'
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = str(random.randint(1024, 65536))
            mp.set_start_method('spawn')
            if world_size == 1:
                _worker_fn(0, world_size, main_fn, args)
            else:
                mp.spawn(
                    _worker_fn,
                    (world_size, main_fn, args),
                    nprocs=world_size,
                    join=True
                )

    return main

def wrap_main_torchrun(main_fn):
    def main(**args):
        local_rank = int(os.environ['LOCAL_RANK'])
        global_rank = int(os.environ['RANK'])
        mp.set_start_method('spawn')
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        main_fn(args)
        dist.destroy_process_group()
    return main

def is_init():
    return dist.is_initialized()

def gather_list(local_list):
    """
    Gather a list from all processes.
    """
    with torch.no_grad():
        if not is_init():
            return local_list
        else:
            gathered_buf = [None for _ in range(world_size())]
            torch.distributed.all_gather_object(gathered_buf, local_list)
            return gathered_buf

def rank():
    if dist.is_initialized():
        return dist.get_rank()
    else:
        return 0

def world_size():
    if dist.is_initialized():
        return dist.get_world_size()
    else:
        return 1

def reduce_sum(x):
    with torch.no_grad():
        if isinstance(x, torch.Tensor):
            x_copy = x.clone()
        else:
            x_copy = torch.tensor(x, device='cuda')
        if dist.is_initialized():
            torch.distributed.all_reduce(
                x_copy, op=torch.distributed.ReduceOp.SUM, async_op=False)
        return x_copy

def reduce_mean(x):
    return reduce_sum(x) / world_size()