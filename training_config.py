import torch
from dataclasses import dataclass

@dataclass
class TrainingParam:

    # distributed data parallel parameters
    ddp_enabled: bool = False
    ddp_rank: int = 0
    ddp_local_rank: int = 0
    ddp_world_size: int = 1
    master_process: bool = True
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # learning rate parameters
    max_lr: float = 5e-4
    min_lr: float = max_lr * 0.1
    num_epoch: int = 32
    
    max_steps: int = 6000 * 32
    warmup_steps: int = 2000
    val_steps: int = 10
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    adam_eps: float = 1e-6
    weight_decay: float = 0.2

    total_batch_size: int = 8196 # 2**13
    micro_batch_size: int = 1024  # micro batch size
    grad_accum_steps: int = 32
