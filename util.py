import torch
import tiktoken
import numpy as np
from torchvision import transforms
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


def get_padding_batch_tensor(token_batch, T):
    max_length = T
    input_ids = []
    attention_masks = []

    for tokens in token_batch:
        padded_tokens = tokens + [0] * (max_length - len(tokens))
        padded_masks = [1] * len(tokens) + [0] * (max_length - len(tokens))

        input_ids.append(padded_tokens)
        attention_masks.append(padded_masks)
    
    return \
        torch.tensor(input_ids, dtype=torch.int64), \
        torch.tensor(attention_masks, dtype=torch.int64)


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


def extract_patches(img_tensor, patch_width):
    C, H, W = img_tensor.shape

    # default image resolution is 224 * 224, patch width is 16
    patches = img_tensor.reshape(
        C,
        H // patch_width, patch_width,
        W // patch_width, patch_width
    ) # [3, 14, 16, 14, 16]

    patches = patches.permute(1, 3, 0, 2, 4) # [14, 14, 3, 16, 16]

    # [14 * 14, 3 * 16 * 16]
    return patches.reshape(-1, 3 * patch_width * patch_width)