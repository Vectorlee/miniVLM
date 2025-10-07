import torch
import tiktoken
import numpy as np
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
from PIL import Image

#initilize tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']

# tokenize an individual document
def tokenize(text):
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


def strip_state_prefix(state_dict, prefix="_orig_mod.module."):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_key = k[len(prefix):]
        else:
            new_key = k
        new_state_dict[new_key] = v
    return new_state_dict