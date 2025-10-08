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
    max_lr = 5e-4
    min_lr = max_lr * 0.1
    num_epoch = 32
    
    max_steps = 6000 * 32
    warmup_steps = 2000
    val_steps = 10
    adam_beta1 = 0.9
    adam_beta2 = 0.98
    adam_eps = 1e-6
    weight_decay = 0.2

    total_batch_size = 8196 # 2**13
    micro_batch_size = 1024  # micro batch size
    grad_accum_steps = 32
