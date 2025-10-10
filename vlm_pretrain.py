from dataclasses import dataclass
import os
import torch
import time
import random
from qwenvl_model import QwenVLConfig, QwenVL
from vlm_dataloader import DataLoaderLite

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


def load_qwenvl_model(train_config: TrainingParam):
    model = QwenVL(QwenVLConfig())
    # freeze the llm backbone, prevent it from pre-training
    model.freeze_llm_backbone()

    model = model.to(train_config.device)

    # compile the model, for kernel fuse
    model.adapter = torch.compile(model.adapter)
    model.vision_encoder = torch.compile(model.vision_encoder)

    if train_config.ddp_enabled:
        model = DDP(model, device_ids=[train_config.ddp_local_rank])    

    optimizer = configure_optimizers(model, train_config)

    return model, optimizer



# ------------ Main Training Code ---------------

train_param = TrainingParam(
    max_lr = 2e-4,
    min_lr = 2e-5,
    num_epoch = 32,
    
    max_steps = 6000,
    warmup_steps = 1000,

    total_batch_size = 8196, # 2**13
    micro_batch_size = 128,  # micro batch size
    grad_accum_steps = 4
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
model, optimizer = load_qwenvl_model(train_param)

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

real_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model

# training loop
for step in range(train_param.max_steps):
    t0 = time.time()

    # checkpoint model
    if step % 500 == 0 or step == train_param.max_steps - 1:
        if train_param.master_process:
            torch.save(real_model.vision_encoder.state_dict(), os.path.join(model_dir, f"vlm_vit_{step}.pth"))
            torch.save(real_model.adapter.state_dict(), os.path.join(model_dir, f"vlm_adapter_{step}.pth"))

    # validation loop
    if step % 200 == 0 or step == train_param.max_steps - 1:
        model.eval()
        
        with torch.no_grad():
            val_loss_accum = 0.0
            for _ in range(train_param.val_steps):
                text, mask, img, labels = val_loader.next_batch()
                text, mask, img, labels = \
                    text.to(train_param.device), mask.to(train_param.device), img.to(train_param.device), labels.to(train_param.device)

                with torch.autocast(device_type=train_param.device, dtype=torch.bfloat16):
                    logits, loss = model(input_ids=text, attention_masks=mask, img_tensor=img, labels=labels)
                
                val_loss_accum += loss.detach() / train_param.val_steps

        if train_param.master_process:
            print(f"Validation loss: {val_loss_accum.item():.6f}")
            with open(log_file, 'a') as f:
                f.write(f"{step} val {val_loss_accum.item():.6f}\n")

    # training loop
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0

    for micro_step in range(train_param.grad_accum_steps):
        text, mask, img, labels = train_loader.next_batch()
        text, mask, img, labels = \
            text.to(train_param.device), mask.to(train_param.device), img.to(train_param.device), labels.to(train_param.device)

        # mixed precision training
        with torch.autocast(device_type=train_param.device, dtype=torch.bfloat16):
            not_last_microstep = micro_step < train_param.grad_accum_steps - 1

            if train_param.ddp_enabled and not_last_microstep:
                with model.no_sync():
                    # no_sync context requires the forward pass also resides in the context
                    logits, loss = model(input_ids=text, attention_masks=mask, img_tensor=img, labels=labels)

                    # the micro batch lost the normalizer
                    # so we divide the loss by the number of micro step count
                    loss = loss / train_param.grad_accum_steps
                    loss_accum += loss.detach()
                    # because we didn't zero the grad, the gradient will accumulate
                    loss.backward()
            else:
                # without the no_sync context manager here
                ogits, loss = model(input_ids=text, attention_masks=mask, img_tensor=img, labels=labels)
                loss = loss / train_param.grad_accum_steps
                loss_accum += loss.detach()
                # For the ddp case, the gradients will be synchronized across devices
                loss.backward()

    if train_param.ddp_enabled:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

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
        print(f"step {step}, loss: {loss_accum.item():.6f}, dt: {dt * 1000:.2f}ms, norm: {norm:.4f}, lr: {lr:e}")
        with open(log_file, 'a') as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")


if train_param.ddp_enabled:
    destroy_process_group()