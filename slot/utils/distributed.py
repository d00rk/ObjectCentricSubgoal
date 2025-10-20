import os
import torch
import torch.distributed as dist


def set_distributed(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    print(f"[INFO] DDP initialized on rank {rank}")


def cleanup_distributed():
    dist.destroy_process_group()