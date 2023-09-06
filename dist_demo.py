import torch
import torch_musa
import torch.nn.parallel
import torch.distributed as dist
from torch.distributed import ReduceOp
import torch.multiprocessing as mp

import os
import argparse

LOCAL_RANK = int(os.environ['LOCAL_RANK'])
WORLD_SIZE = int(os.environ['WORLD_SIZE'])
WORLD_RANK = int(os.environ['RANK'])
print(WORLD_RANK)
# os.environ['MUSA_VISIBLE_DEVICES'] = f'{WORLD_RANK + 1}'

def setup():
    # initialize the process group
    dist.init_process_group("mccl", rank=WORLD_RANK, world_size=WORLD_SIZE)

def cleanup():
    dist.destroy_process_group()

def test_allgather():
    # print(f"Running basic DDP example on rank {rank}.")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print("my rank = %d my size = %d" % (rank, world_size))
    torch_musa.set_device(rank)
    device = torch.device(f'musa')

    tensor_list = [torch.zeros(100, dtype=torch.float32).to(device) for _ in range(world_size)]
    tensor = torch.arange(100, dtype=torch.float32).to(device) + 1 + 2 * rank

    print(rank, tensor)
    # dist.reduce_scatter(tensor, tensor_list)
    print(rank, tensor_list)
    dist.all_gather(tensor_list, tensor)
    print(rank, tensor_list)

def test_broacast():
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print("my rank = %d my size = %d" % (rank, world_size))
    device = torch.device(f'musa:{rank}')
    if rank == 0:
        tensor = torch.randn(4, 5).to(device)
    else:
        tensor = torch.zeros(4, 5).to(device)
    dist.broadcast(tensor, 0)
    print(rank, tensor)

def test_allreduce():
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print("my rank = %d my size = %d" % (rank, world_size))
    device = torch.device(f'musa:{rank}')
    #device = torch.device('cpu')
    tensor = torch.arange(2, dtype=torch.float32).to(device) + 1 + 2 * rank
    print(tensor)
    dist.all_reduce(tensor)#, op=ReduceOp.AVG)
    # dist.all_reduce(tensor)
    print(tensor)

def test_reduce():
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print("my rank = %d my size = %d" % (rank, world_size))
    device = torch.device(f'musa:{rank}')

    tensor = torch.arange(2, dtype=torch.float32).to(device) + 1 + 2 * rank
    print(tensor)
    dist.reduce(tensor, dst=0, op=ReduceOp.AVG)
    # dist.all_reduce(tensor)
    print(tensor)

def test_gather():
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print("my rank = %d my size = %d" % (rank, world_size))
    device = torch.device(f'musa:{rank}')

    tensor_list = [torch.zeros(100, dtype=torch.float32).to(device) for _ in range(world_size)]
    tensor = torch.arange(100, dtype=torch.float32).to(device) + 1 + 2 * rank

    # dist.reduce_scatter(tensor, tensor_list)
    print(rank, tensor_list)
    dist.gather(tensor, tensor_list, dst=1)
    print(rank, tensor_list)

def test_scatter():
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print("my rank = %d my size = %d" % (rank, world_size))
    device = torch.device(f'musa:{rank}')

    tensor_size = 10
    t_ones = torch.ones(tensor_size)
    t_fives = torch.ones(tensor_size) * 5
    output_tensor = torch.zeros(tensor_size)
    if dist.get_rank() == 0:
        # Assumes world_size of 2.
        # # Only tensors, all of which must be the same size.
        scatter_list = [t_ones, t_fives]
    else:
        scatter_list = None
    dist.scatter(output_tensor, scatter_list, src=0)
    # Rank i gets scatter_list[i]. For example, on rank 1:
    print(output_tensor)

def test_reducescatter():
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print("my rank = %d my size = %d" % (rank, world_size))
    torch_musa.set_device(rank)
    device = torch.device(f'musa')
    print(device)
    tensor_size = 10
    input_tensor_list = [torch.arange(tensor_size, dtype=torch.float32).to(device) + 1 + 2 * rank for _ in range(world_size)]
    output = torch.zeros(tensor_size).to(device)
    print(input_tensor_list)

    print(rank, output)
    dist.reduce_scatter(output, input_tensor_list)
    print(rank, output)

def test_barrier():
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    device = torch.device(f'musa:{rank}')
    if rank == 0:
        import time
        time.sleep(10)
    print("my rank = %d my size = %d" % (rank, world_size))

def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)
    
def test_function():
    setup()

    # test_broacast()
    # test_allgather()
    # test_allreduce()
    # test_gather()
    # test_scatter()
    # test_barrier()
    test_reducescatter()
    cleanup()

if __name__ == "__main__":
    # run_demo(demo_basic, 2)
    test_function()

