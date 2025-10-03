import os
import torch
import numpy as np
import random
import tiktoken
from dataclasses import dataclass
from util import load_image, extract_patches, tokenize, get_padding_batch_tensor

DATA_ROOT = "clip_data/"
INPUT_KEY = "input_ids"
MASK_KEY = "attention_masks"
PATCH_KEY = "image_patches"


class DataLoaderLite:

    def __init__(self, batch_size, process_rank, num_processes, split):
        self.batch_size = batch_size
        self.process_rank = process_rank
        self.num_processes = num_processes

        self.current_shard = process_rank
        self.current_data = {}
        self.data_index = 0
        assert split in ('train', 'val')

        # list the file
        data_root = DATA_ROOT
        shards = os.listdir(data_root)
        if split == 'val':
            shards = [s for s in shards if s.startswith("val_")]
        else:
            shards = [s for s in shards if not s.startswith("val_")]
        
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"No shards found for {split} under {data_root}"
        print(f"Found {len(shards)} shards for split {split}")

        self.reset()


    def reset(self):
        # each process starts from their corresponding index
        self.current_shard = self.process_rank
        self.data_index = 0

        data = torch.load(self.shards[self.current_shard])
        assert isinstance(data, dict)
        assert (INPUT_KEY in data) and (MASK_KEY in data) and (PATCH_KEY in data)
        
        self.current_data = data
        return
    

    def get_data(self, tensor_data, start, end):
        caption_input_ids = tensor_data[INPUT_KEY][start: end]
        caption_attention_masks = tensor_data[MASK_KEY][start: end]
        image_patches = tensor_data[PATCH_KEY][start: end]

        return caption_input_ids, caption_attention_masks, image_patches


    def next_batch(self):

        caption_input_ids, caption_attention_masks, image_patches = \
            self.get_data(self.current_data, self.data_index, self.data_index + self.batch_size)
        
        if self.data_index + self.batch_size > len(self.current_data[PATCH_KEY]):
            self.data_index = self.data_index + self.batch_size - len(self.current_data[PATCH_KEY])

            # load new data shard, each process will load its corresponding next shard
            self.current_shard = (self.current_shard + self.num_processes) % len(self.shards)
            self.current_data = torch.load(self.shards[self.current_shard])

            new_input_ids, new_attention_masks, new_image_patches = \
                self.get_data(self.current_data, 0, self.data_index)
            
            caption_input_ids = torch.stack(caption_input_ids, new_input_ids)
            caption_attention_masks = torch.stack(caption_attention_masks, new_attention_masks)
            image_patches = torch.stack(image_patches, new_image_patches)
            
        else:
            self.data_index = self.data_index + self.batch_size

        return caption_input_ids, caption_attention_masks, image_patches