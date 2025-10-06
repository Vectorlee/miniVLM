from dataclasses import dataclass
import os
import torch
import math
import time
import random
from clip_dataloader import DataLoaderLite
from clip_model import CLIPModel, CLIPConfig

import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP


# set the random seed to ensure reproducibility
random.seed(1337)
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)


@dataclass
class CLIPTrainingParam:
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
    
    max_steps = 1800
    warmup_steps = 600
    adam_beta1 = 0.9
    adam_beta2 = 0.98
    adam_eps = 1e-6
    weight_decay = 0.2

    total_batch_size = 32768 # 2**15
    micro_batch_size = 64  # micro batch size
    grad_accum_steps = 64



def config_ddp(train_config: CLIPTrainingParam):
    # Set up DDP (Distributed Data Parallel)
    # torchrun creates these environment variables
    ddp = int(os.environ.get('RANK', -1)) != -1 
    if ddp:
        train_config.ddp_enabled = True

        train_config.ddp_rank = int(os.environ['RANK'])
        train_config.ddp_local_rank = int(os.environ['LOCAL_RANK'])
        train_config.ddp_world_size = int(os.environ['WORLD_SIZE'])
        train_config.device = f'cuda:{train_config.ddp_local_rank}'

        # now we need cuda
        assert torch.cuda.is_available(), "Need CUDA for DDP"        
        init_process_group(
            backend = "nccl",
            world_size = train_config.ddp_world_size,
            rank = train_config.ddp_local_rank
        )
        
        train_config.master_process = train_config.ddp_rank == 0
    else:
        # Vanilla
        train_config.ddp_enabled = False
        train_config.ddp_rank = 0
        train_config.ddp_local_rank = 0
        train_config.ddp_world_size = 1
        train_config.master_process = True
        train_config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    return train_config



def get_lr(it, train_config: CLIPTrainingParam):
    # linear warmup for warmup_iters steps
    if it < train_config.warmup_steps:
        return train_config.max_lr * (it + 1) / train_config.warmup_steps
    # if iter > max_steps, use the costant min learing rate
    if it > train_config.max_steps:
        return train_config.min_lr

    # cosine decay down to min learning rate
    decay_ratio = (it - train_config.warmup_steps) / \
        (train_config.max_steps - train_config.warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return train_config.min_lr + coeff * (train_config.max_lr - train_config.min_lr)



def configure_optimizers(model, train_config: CLIPTrainingParam):
    # start with all of the parameters that require grad
    param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    # do not weight decay bias, layernorm, and other less than 2 dimension weights
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": train_config.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    if train_config.master_process:
        print(f"num decayed tensors: {len(decay_params)}, with {num_decay_params} parameters")
        print(f"num non-decayed tensors: {len(nodecay_params)}, with {num_nodecay_params} parameters")

    fused = True if torch.cuda.is_available() else False
    optimizer = torch.optim.AdamW(
        optim_groups, 
        lr = train_config.min_lr,
        betas = (train_config.adam_beta1, train_config.adam_beta2),
        eps = train_config.adam_eps,
        fused = fused
    )
    return optimizer



def load_clip_model(train_config: CLIPTrainingParam):
    model = CLIPModel(CLIPConfig())

    # Recommended order: 
    #   1. Move the model to the target device
    #   2. Wrap the model in ddp with the correct local rank id
    #   3. Compile the model
    #   4. Create the optimizer object
    model = model.to(train_config.device)

    if train_config.ddp_enabled:
        model = DDP(model, device_ids=[train_config.ddp_local_rank])

    # compile the model, for kernel fuse
    model = torch.compile(model)

    optimizer = configure_optimizers(model, train_config)

    return model, optimizer


# ------------ Main Training Code ---------------

train_param = CLIPTrainingParam()

# config DDP settings
train_param = config_ddp(train_param)
# set up grad accum step
train_param.grad_accum_steps = \
    train_param.total_batch_size // train_param.ddp_world_size // train_param.micro_batch_size

if train_param.master_process:
    print(f"Is DDP enabled: {train_param.ddp}")
    print(f"DDP word_size: {train_param.ddp_world_size}, DDP rank: {train_param.ddp_rank}")
    print(f"Device using: {train_param.device}")
    print(f"Total desired batch size: {train_param.total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {train_param.grad_accum_steps}")

torch.cuda.set_device(train_param.device)
torch.set_float32_matmul_precision('high')

# load model
model, optimizer = load_clip_model(train_param)

# set up logging
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log_file.txt")
with open(log_file, "w") as f:
    pass

model_dir = "model"
os.makedirs(model_dir, exist_ok=True)


# use the micro batch size in the data loader
train_loader = DataLoaderLite(
    batch_size = train_param.micro_batch_size, 
    split = 'train'
)
val_loader = DataLoaderLite(
    batch_size = train_param.micro_batch_size,
    split = 'val'
)

# training loop
for step in range(train_param.max_steps):
    t0 = time.time()

    # checkpoint model
    # if step % 10000 == 0 or step == train_param.max_steps - 1:
    #    if train_param.master_process:
    #        torch.save(model.state_dict(), os.path.join(model_dir, f"model_{step}.pth"))

    # validation loop
    if step % 500 == 0 or step == train_param.max_steps - 1:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            # need to be smaller than 100M val tokens / (B * T * world_size)
            # roughly processing 10M tokens for validation
            val_loss_steps = 40 
            for _ in range(val_loss_steps):
                text, mask, img = val_loader.next_batch()
                text, mask, img = text.to(train_param.device), mask.to(train_param.device), img.to(train_param.device)

                with torch.autocast(device_type=train_param.device, dtype=torch.bfloat16):
                    logit, loss = model(input_ids = text, attention_masks = mask, img_tensor = img)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        
        if train_param.ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if train_param.master_process:
            print(f"Validation loss: {val_loss_accum.item():.6f}")
            with open(log_file, 'a') as f:
                f.write(f"{step} val {val_loss_accum.item():.6f}\n")

    # training loop
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0

    for micro_step in range(train_param.grad_accum_steps):
        text, mask, img = train_loader.next_batch()
        text, mask, img = text.to(train_param.device), mask.to(train_param.device), img.to(train_param.device)

        # mixed precision training
        with torch.autocast(device_type=train_param.device, dtype=torch.bfloat16):
            not_last_microstep = micro_step < train_param.grad_accum_steps - 1

            if train_param.ddp and not_last_microstep:
                with model.no_sync():
                    # no_sync context requires the forward pass also resides in the context
                    logits, loss = model(input_ids = text, attention_masks = mask, img_tensor = img)

                    # the micro batch lost the normalizer
                    # so we divide the loss by the number of micro step count
                    loss = loss / train_param.grad_accum_steps
                    loss_accum += loss.detach()
                    # because we didn't zero the grad, the gradient will accumulate
                    loss.backward()
            else:
                # without the no_sync context manager here
                logits, loss = model(input_ids = text, attention_masks = mask, img_tensor = img)
                loss = loss / train_param.grad_accum_steps
                loss_accum += loss.detach()
                # For the ddp case, the gradients will be synchronized across devices
                loss.backward()

    if train_param.ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

    # Gradient Clipping
    # Before the optimizer.step, but after the loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # set the cosine decay learing rate
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()

    t1 = time.time()
    dt = (t1 - t0)
    if train_param.master_process:
        print(f"step {step}, loss: {loss_accum.item():.6f}, dt: {dt * 1000:.2f}ms, norm: {norm:.4f}, lr: {lr:e}")
        with open(log_file, 'a') as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")


if train_param.ddp:
    destroy_process_group()