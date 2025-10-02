import os
from datetime import timedelta
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# basic rule of func:
#   1st argument: process index
#   2nd argument: collection of args, including addr, port and num_gpu
def run_process(func, args):
    ''' Problem 1: Run process
    (./handler/DDP/utils.py)
    Implement run_process function.
    run_process function is used to run main_func in multiple processes, for DDP GPU group.
    It is a wrapper of mp.spawn function.
    You can use mp.spawn function as a reference.
    '''
    
    nprocs = args.num_gpu
    assert nprocs > 0, "num_gpu must be > 0"
    assert torch.cuda.is_available(), "CUDA is required for DDP (nccl backend)."
    mp.spawn(
        fn=func,               # 실행할 함수
        args=(args,),          # main_func의 두 번째 인자 (args)
        nprocs=nprocs,         # 실행할 프로세스 수
        join=True              # 모든 프로세스 종료까지 대기
    )

def initialize_group(proc_id, host, port, num_gpu):
    ''' Problem 2: Setup GPU group
    (./handler/DDP/utils.py)
    DDP requires to setup GPU group, which can broadcast weights to all GPUs.
    This function set tcp connection between processes.
    Implement initialize_group function.

    you should use
    1. dist.init_process_group() for tcp connection
    2. torch.cuda.set_device() for setting device
    '''

    torch.cuda.set_target_device(proc_id) if hasattr(torch.cuda, "set_target_device") else None
    torch.cuda.set_device(proc_id)

    dist_url = f"tcp://{host}:{port}"
    dist.init_process_group(
        backend="nccl",                 # GPU면 nccl
        init_method=dist_url,           # 'env://'도 가능하지만, 문제 요구가 tcp라서 tcp 사용
        world_size=num_gpu,
        rank=proc_id,
        timeout=timedelta(minutes=10),
    )

    # 선택: 모두 준비될 때까지 대기(동기화 지점)
    dist.barrier()

def destroy_process():
    ''' Problem 3: Destroy GPU group
    (./handler/DDP/utils.py)
    Implement destroy_process function.
    Just call the torch.distributed's destroy function.
    '''
    if dist.is_initialized():
        dist.destroy_process_group()

