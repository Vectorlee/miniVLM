import webdataset as wds
from torchvision import transforms
import tiktoken
import torch
import torch.distributed as dist
import glob

from util import get_padding_batch_input, gpt_tokenize


TRAIN_DATA_PATTERN = "clip_data/training_data/*.tar"
VAL_DATA_PATTERN = "clip_data/validation_data/*.tar"

TEXT_LENGTH_LIMIT = 128

def create_dataloader(shard_pattern, batch_size):
    shard_urls = sorted(glob.glob(shard_pattern))
    print(f"creating dataloader, shard_pattern: {shard_pattern}, shard_count: {len(shard_urls)}")
    assert len(shard_urls) > 0

    image_transform = transforms.Compose([
        # the image has already been resize by img2dataset
        transforms.ToTensor(),           # convert to tensor [C, H, W] in [0,1]
        transforms.Normalize(            # normalize with mean/std
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

    def caption_transform(json_data):
        if not json_data['caption']:
            return []

        caption = json_data['caption']
        tokens = gpt_tokenize(caption)
        return tokens

    def filter_sample(sample):
        _, caption_token = sample
        return len(caption_token) < TEXT_LENGTH_LIMIT and len(caption_token) > 1

    splitter_func = wds.split_by_node if dist.is_initialized() else wds.single_node_only
    dataset = (
        wds.WebDataset(
            shard_urls,
            nodesplitter = splitter_func,
            handler = wds.ignore_and_continue,
            resampled = True,
            shardshuffle = True)
        .shuffle(1000)  # sample-level shuffle buffer
        .decode("pil")  # decode jpg->PIL
        .to_tuple("jpg", "json")
        .map_tuple(image_transform, caption_transform)
        .select(filter_sample)
        .batched(batch_size)
    )

    # To know which data shard we are fetching from, we can do:
    #   dataset = wds.WebDataset(url).to_tuple("jpg", "json", "__url__")
    #   img, json_data, url = next(iter(dataset))

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        num_workers=4,
    )

    return loader


class DataLoaderLite:

    def __init__(self, batch_size, split):
        assert split in ('train', 'val')
        
        self.batch_size = batch_size
        self.shard_pattern = TRAIN_DATA_PATTERN if split == 'train' else VAL_DATA_PATTERN
        
        self.dataloader_iter = iter(create_dataloader(self.shard_pattern, batch_size))

    def next_batch(self):
        img_batch, token_batch = next(self.dataloader_iter)
        input_ids, attention_masks = get_padding_batch_input(token_batch)

        return input_ids, attention_masks, img_batch


