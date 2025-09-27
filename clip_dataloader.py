import os
import torch
import numpy as np
import random
import tiktoken
from dataclasses import dataclass
from util import load_image, extract_patches, tokenize, get_padding_batch_tensor

DATA_ROOT = "pretrain_data"

@dataclass
class ClipDataConfig:
    img_resolution: int = 224
    patch_dimension: int = 16
    batch_size: int = 4096
    caption_length: int = 128


class DataLoaderLite:

    def __init__(self, config, process_rank, num_processes, split):
        self.B = config.batch_size
        self.T = config.caption_length
        self.resolution = config.img_resolution
        self.patch_width = config.patch_dimension

        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in ('train', 'val')

        # list the file
        data_root = DATA_ROOT
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"No shards found for {split} under {data_root}"
        print(f"Found {len(shards)} shards for split {split}")

        self.reset()


    def reset(self):
        # set the starting shard randomly to create some randomness in data loading
        self.current_shard = random.randint(0, len(self.shards)) % len(self.shards)
        pass

    def next_batch(self):

        # load image and their captions
        # PLACEHOLDER
        image_file_list = [""] * self.B
        caption_list = [""] * self.B

        image_tensors = [load_image(filename, self.resolution) for filename in image_file_list]

        # [14 * 14, 3 * 16 * 16] * B
        batch_patches = [extract_patches(x, self.patch_width) for x in image_tensors]
        batch_patches = torch.stack(batch_patches, dim=0) # [B, 14 * 14, 3 * 16 * 16]

        caption_tokens = [tokenize(txt) for txt in caption_list]
        caption_input_ids, caption_attention_mask = get_padding_batch_tensor(caption_list, self.T)


        return batch_patches, caption_input_ids, caption_attention_mask