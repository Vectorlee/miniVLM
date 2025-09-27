import os
import torch
import math
import time
import random
from clip_dataloader import DataLoaderLite
from model import GPT, GPTConfig
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

# Set up DDP (Distributed Data Parallel)
# torchrun creates these environment variables
ddp = int(os.environ.get('RANK', -1)) != -1 
if ddp:
    # now we need cuda
    assert torch.cuda.is_available(), "Need CUDA for DDP"
    init_process_group(backend="nccl")

    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    # Vanilla
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

# set the random seed to ensure reproducibility
random.seed(1337)
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# learning rate scheduler
max_lr = 6e-4
min_lr = max_lr * 0.1
# warmup_steps = 10
# max_steps = 50
# For 10B tokens, divided by step size 524288, training 1 epoch
max_steps = 19073
warmup_steps = 715


def get_lr(it):
    # linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    # if iter > max_steps, use the costant min learing rate
    if it > max_steps:
        return min_lr

    # cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def configure_optimizers(model, weight_decay, learning_rate):
    # start with all of the parameters that require grad
    param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    # do not weight decay bias, layernorm, and other less than 2 dimension weights
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"num decayed tensors: {len(decay_params)}, with {num_decay_params} parameters")
    print(f"num non-decayed tensors: {len(nodecay_params)}, with {num_nodecay_params} parameters")

    fused = True if torch.cuda.is_available() else False
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=fused)
    return optimizer

# gradient accumulate
# following the GPT-3 paper 0.5M batch size setting
total_batch_size = 524288 # 2**19, in number of tokens, nice number
B = 32 # micro batch size
T = 1024 # sequence length

# divisible
assert total_batch_size % (B * T * ddp_world_size) == 0
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

if master_process:
    print(f"Is DDP enabled: {ddp}")
    print(f"DDP word_size: {ddp_world_size}, DDP rank: {ddp_rank}")
    print(f"Device using: {device}")
    print(f"Total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

torch.set_float32_matmul_precision('high')


# Change the original 50257 token count into a nice number
# Nice numbers are the numbers that can be divided by large power of 2 numbers
config = GPTConfig(vocab_size=50304)
model = GPT(config)

# Recommended order: 
#   1. Move the model to the target device
#   2. Wrap the model in ddp with the correct local rank id
#   3. Compile the model
#   4. Create the optimizer object
model = model.to(device)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# compile the model, for kernel fuse
model = torch.compile(model)

#optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = configure_optimizers(model, weight_decay=0.1, learning_rate=6e-4)

# use the micro batch size in the data loader
train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='train')
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='val')

log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log_file.txt")
with open(log_file, "w") as f:
    pass

model_dir = "model"
os.makedirs(model_dir, exist_ok=True)

for step in range(max_steps):
    t0 = time.time()

    # checkpoint model
    if step % 1000 == 0 or step == max_steps - 1:
        if master_process:
            torch.save(model.state_dict(), os.path.join(model_dir, f"model_{step}.pth"))

    # validation loop
    if step % 100 == 0 or step == max_steps - 1:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            # need to be smaller than 100M val tokens / (B * T * world_size)
            # roughly processing 10M tokens for validation
            val_loss_steps = 40 
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logit, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"Validation loss: {val_loss_accum.item():.6f}")
            with open(log_file, 'a') as f:
                f.write(f"{step} val {val_loss_accum.item():.6f}\n")

    # training loop
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0

    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)

        # mixed precision training
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            not_last_microstep = micro_step < grad_accum_steps - 1

            if ddp and not_last_microstep:
                with model.no_sync():
                    # no_sync context requires the forward pass also resides in the context
                    logits, loss = model(x, y)

                    # the micro batch lost the normalizer
                    # so we divide the loss by the number of micro step count
                    loss = loss / grad_accum_steps
                    loss_accum += loss.detach()
                    # because we didn't zero the grad, the gradient will accumulate
                    loss.backward()
            else:
                # without the no_sync context manager here
                logits, loss = model(x, y)
                loss = loss / grad_accum_steps
                loss_accum += loss.detach()
                # For the ddp case, the gradients will be synchronized across devices
                loss.backward()

    if ddp:
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
    token_processed = B * T * grad_accum_steps * ddp_world_size
    token_per_sec = token_processed / dt
    # the item() function ship the tensor back from gpu to cpu
    if master_process:
        print(f"step {step}, loss: {loss_accum.item():.6f}, dt: {dt * 1000:.2f}ms, tok/sec: {token_per_sec}, norm: {norm:.4f}, lr: {lr:e}")
        with open(log_file, 'a') as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")

if ddp:
    destroy_process_group()