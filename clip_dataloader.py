import webdataset as wds
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
import tiktoken
import torch
import torch.distributed as dist

TRAIN_DATA_PATTERN = "clip_data/training_data/{00000..11499}.tar"
VAL_DATA_PATTERN = "clip_data/validation_data/{00000..0031}.tar"

#initilize tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']

def tokenize(text):
    tokens = []
    tokens.extend(enc.encode_ordinary(text))
    tokens.append(eot)
    return tokens


def create_dataloader(shard_pattern, batch_size):

    image_transform = transforms.Compose([
        # the image has already been resize by img2dataset
        transforms.ToTensor(),           # convert to tensor [C, H, W] in [0,1]
        transforms.Normalize(            # normalize with mean/std
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

    def caption_transform(json_data):
        caption = json_data['caption']
        tokens = tokenize(caption)
        return tokens

    def filter_sample(sample):
        _, caption_token = sample
        return len(caption_token) < 128

    dataset = (
        wds.WebDataset(shard_pattern, resampled=True, shardshuffle=True)
        .shuffle(1000)  # sample-level shuffle buffer
        .decode("pil")  # decode jpg->PIL
        .to_tuple("jpg", "json")
        .map_tuple(image_transform, caption_transform)
        .select(filter_sample)
        .batched(batch_size)
    )

    # shard dataset across ranks
    if dist.is_initialized():
        dataset = dataset.shard(dist.get_world_size(), dist.get_rank())

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        num_workers=4,
    )

    return loader


def get_padding_batch_input(token_batch):
    input_list = []
    mask_list = []

    for tokens in token_batch:
        input_list.append(torch.tensor(tokens, dtype=torch.int64))
        mask_list.append(torch.ones(len(tokens), dtype=torch.int64))
    
    input_ids = pad_sequence(input_list, batch_first=True)
    attention_masks = pad_sequence(mask_list, batch_first=True)
    
    return input_ids, attention_masks


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


