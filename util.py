import os
import math
import tiktoken
import numpy as np
import torch
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
from torch.distributed import init_process_group
from PIL import Image

from training_config import TrainingParam

#initilize tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']

# tokenize an individual document
def gpt_tokenize(text):
    tokens = []
    tokens.extend(enc.encode_ordinary(text))
    tokens.append(eot)
    return tokens


def get_padding_batch_input(token_batch):
    input_list = []
    mask_list = []

    for tokens in token_batch:
        input_list.append(torch.tensor(tokens, dtype=torch.int64))
        mask_list.append(torch.ones(len(tokens), dtype=torch.int64))
    
    input_ids = pad_sequence(input_list, batch_first=True)
    attention_masks = pad_sequence(mask_list, batch_first=True)
    
    return input_ids, attention_masks


def load_image(filename, resolution):
    img = Image.open(filename).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),           # convert to tensor [C, H, W] in [0,1]
        transforms.Normalize(            # normalize with mean/std
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

    img_tensor = transform(img)
    return img_tensor


def strip_state_prefix(state_dict, custom_prefix="_orig_mod.module."):
    ddp_prefix = "_orig_mod.module."
    regular_prefix = "_orig_mod."

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(ddp_prefix):
            new_key = k[len(ddp_prefix):]
        elif k.startswith(regular_prefix):
            new_key = k[len(regular_prefix):]
        elif k.startswith(custom_prefix):
            new_key = k[len(custom_prefix):]
        else:
            new_key = k
        new_state_dict[new_key] = v
    return new_state_dict


# Training utility methods:

def configure_optimizers(model, train_config: TrainingParam):
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


def get_lr(it, train_config: TrainingParam):
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



def config_ddp(train_config: TrainingParam):
    # Set up DDP (Distributed Data Parallel)
    # torchrun creates these environment variables
    ddp = int(os.environ.get('RANK', -1)) != -1 
    if ddp:
        # now we need cuda
        assert torch.cuda.is_available(), "Need CUDA for DDP"
        train_config.ddp_enabled = True

        train_config.ddp_rank = int(os.environ['RANK'])
        train_config.ddp_local_rank = int(os.environ['LOCAL_RANK'])
        train_config.ddp_world_size = int(os.environ['WORLD_SIZE'])
        train_config.device = f'cuda:{train_config.ddp_local_rank}'

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