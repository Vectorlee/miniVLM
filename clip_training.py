from dataclasses import dataclass
import os
import torch
import math
import time
import random
from clip_dataloader import DataLoaderLite
from clip_model import CLIPModel, clip_loss
from vision_transformer import VisionTransformerConfig
from text_encoder import TextEncoderConfig

import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from training_config import TrainingParam
from util import configure_optimizers, get_lr, config_ddp

# set the random seed to ensure reproducibility
random.seed(1337)
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)


def load_clip_model(train_config: TrainingParam):
    clip_model_hidden_size = 512

    model = CLIPModel(
        text_config = TextEncoderConfig(),
        vision_config = VisionTransformerConfig(),
        hidden_dim = clip_model_hidden_size
    )

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


# Gathers tensors from all ranks and supports gradient backprop.
def all_gather_with_grad(tensor):
    if not dist.is_initialized():
        return tensor

    world_size = dist.get_world_size()
    tensors_gather = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensors_gather, tensor)

    # ensure local copy is preserved for autograd
    tensors_gather[dist.get_rank()] = tensor
    return torch.cat(tensors_gather, dim=0)


# ------------ Main Training Code ---------------

train_param = TrainingParam(
    max_lr = 5e-4,
    min_lr = 5e-5,
    num_epoch = 32,
    
    max_steps = 6000 * 32,
    warmup_steps = 2000,
    val_steps = 10,

    total_batch_size = 8196, # 2**13
    micro_batch_size = 1024,  # micro batch size
    grad_accum_steps = 1
)

# config DDP settings
train_param = config_ddp(train_param)

print(f"Is DDP enabled: {train_param.ddp_enabled}")
print(f"DDP word_size: {train_param.ddp_world_size}, DDP rank: {train_param.ddp_rank}")
print(f"Device using: {train_param.device}")

# set device
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
    if step % 3000 == 0 or step == train_param.max_steps - 1:
        if train_param.master_process:
            torch.save(model.state_dict(), os.path.join(model_dir, f"model_{step}.pth"))

    # validation loop
    if step % 500 == 0 or step == train_param.max_steps - 1:
        model.eval()
        
        with torch.no_grad():
            val_loss_accum = 0.0
            for _ in range(train_param.val_steps):
                text, mask, img = val_loader.next_batch()
                text, mask, img = text.to(train_param.device), mask.to(train_param.device), img.to(train_param.device)

                with torch.autocast(device_type=train_param.device, dtype=torch.bfloat16):
                    text_embds, vision_embds, scaler = model(input_ids = text, attention_masks = mask, img_tensor = img)
                    # get embeddings from other GPUs
                    global_text_embds = all_gather_with_grad(text_embds)
                    global_vision_embds = all_gather_with_grad(vision_embds)

                    loss = clip_loss(global_text_embds, global_vision_embds, scaler)
                
                val_loss_accum += loss.detach() / train_param.val_steps

        if train_param.master_process:
            print(f"Validation loss: {val_loss_accum.item():.6f}")
            with open(log_file, 'a') as f:
                f.write(f"{step} val {val_loss_accum.item():.6f}\n")

    # training loop
    model.train()
    optimizer.zero_grad()

    text, mask, img = train_loader.next_batch()
    text, mask, img = text.to(train_param.device), mask.to(train_param.device), img.to(train_param.device)

    # mixed precision training
    with torch.autocast(device_type=train_param.device, dtype=torch.bfloat16):
        text_embds, vision_embds, scaler = model(input_ids = text, attention_masks = mask, img_tensor = img)

        # get embeddings from other GPUs
        global_text_embds = all_gather_with_grad(text_embds)
        global_vision_embds = all_gather_with_grad(vision_embds)
            
        loss = clip_loss(global_text_embds, global_vision_embds, scaler)
        loss.backward()

    # Gradient Clipping
    # Before the optimizer.step, but after the loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # set the cosine decay learing rate
    lr = get_lr(step, train_param)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()

    t1 = time.time()
    dt = (t1 - t0)
    if train_param.master_process:
        print(f"step {step}, loss: {loss.item():.6f}, dt: {dt * 1000:.2f}ms, norm: {norm:.4f}, lr: {lr:e}")
        with open(log_file, 'a') as f:
            f.write(f"{step} train {loss.item():.6f}\n")


if train_param.ddp_enabled:
    destroy_process_group()