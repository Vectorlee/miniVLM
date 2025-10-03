from langdetect import detect
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
from multiprocessing import Pool
from tqdm import tqdm
from PIL import Image
import subprocess
import tiktoken
import torch
import json
import os

"""
# First use the following bash script to untar the img2dataset dataset

for f in *.tar; do
    dirname="${f%.tar}_dir"     # remove .tar and add _dir
    mkdir -p "$dirname"         # create directory
    mv "$f" "$dirname"/         # move tar into directory
    tar -xf "$dirname/$f" -C "$dirname"  # extract inside directory
    rm "$dirname/$f"            # remove the tar file
done
"""

INPUT_KEY = "input_ids"
MASK_KEY = "attention_masks"
PATCH_KEY = "image_patches"

enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']

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


def load_image(filename):
    img = Image.open(filename).convert("RGB")

    transform = transforms.Compose([
        # the image has already been resize by img2dataset
        transforms.ToTensor(),           # convert to tensor [C, H, W] in [0,1]
        transforms.Normalize(            # normalize with mean/std
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

    img_tensor = transform(img)
    return img_tensor


def extract_image_patches(img_tensor, patch_width):
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


def convert_data_into_tensor(file_dir, json_file_list, output_file):
    
    text_token_list = []
    img_patch_list = []
    
    for json_file in json_file_list:
        with open(os.path.join(file_dir, json_file)) as pfile:
            data = json.load(pfile)
            
            # skip download failure cases
            if data['status'] != 'success':
                continue
            
            key = data['key']
            caption = data['caption']
            
            # skip non-English captions
            if detect(caption) != 'en':
                continue
            
            # skip over size captions
            tokens = tokenize(caption)
            if len(tokens) >= 128:
                continue
            

            img_file = key + ".jpg"
            img_tensor = load_image(os.path.join(file_dir, img_file))
            img_patches = extract_image_patches(img_tensor, 16)
            
            text_token_list.append(tokens)
            img_patch_list.append(img_patches)
    
    # convert the input into padded tensor
    input_ids, attention_masks = get_padding_batch_input(text_token_list)
    # covert list into tensor
    image_patches = torch.tensor(img_patch_list)

    torch.save({
        INPUT_KEY: input_ids,
        MASK_KEY:  attention_masks,
        PATCH_KEY: image_patches
    }, output_file)

    # return the size of this batch
    return len(img_patch_list)



def prepare_training_data(tar_dir):
    
    file_list = os.listdir(tar_dir)
    json_file_list = list(filter(lambda s: s.endswith(".json"), file_list))
    output_file = tar_dir + "_data.pth"

    # the output file will be written in the current directory
    element_count =  convert_data_into_tensor(tar_dir, json_file_list, output_file)
    return (element_count, output_file)


if __name__ == "__main__":
    tar_suffix = "_tar"

    # list current directory
    tar_dir_list = os.listdir(".")
    tar_dir_list = list(filter(lambda s: s.endswith(tar_suffix), tar_dir_list))

    # the file torch pth file will be stored in the current directory
    data_file_list = []
    total_data_count = 0
    worker = os.process_cpu_count()
    with Pool(processes=worker) as pool:
        for item_count, data_file in tqdm(pool.imap_unordered(prepare_training_data, tar_dir_list, chunksize=4)):
            data_file_list.append(data_file)
            
            total_data_count += item_count
            print(f"Data element count: {item_count}")


    data_file_list = sorted(data_file_list)
    val_file = data_file_list[0]

    subprocess.run(["mv", f"{val_file}", f"val_{val_file}"], shell=True)
    print(f"Total element count: {total_data_count}")

    


