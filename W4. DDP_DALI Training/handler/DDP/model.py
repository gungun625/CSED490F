import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def model_to_DDP(model):
    ''' Problem 4: model to DDP
    (./handler/DDP/model.py)
    Implement model_to_DDP function.
    model_to_DDP function is used to transfer model to DDP, SIMILAR with DP.
    Be careful for set devices. Set profer device id is important part in DDP.
    '''
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DDP.")
    if not dist.is_initialized():
        raise RuntimeError("Process group is not initialized. Call initialize_group first.")

    # initialize_group에서 set_device(proc_id)를 했다면, 현재 디바이스가 곧 local rank
    local_rank = torch.cuda.current_device()  # ex) 0,1,2,3 ...
    device = torch.device("cuda", local_rank)

    # 모델을 해당 GPU로 이동
    model = model.to(device, non_blocking=True)

    # (선택) 배치노름 동기화가 필요하면 활성화
    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # GPU당 1프로세스 기준의 정석 설정
    ddp_model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,   # 필요 시 True
        broadcast_buffers=False         # BN 버퍼 브로드캐스트 안 쓰면 False가 보통 빠름
        # static_graph=True              # 그래프가 고정이면 PyTorch>=2.0에서 이 옵션도 고려
    )
    return ddp_model