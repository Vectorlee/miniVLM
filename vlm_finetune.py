import json
import torch
import torch.nn.functional as F
import random
import time
import os
from multiprocessing import Pool

from qwenvl_model import QwenVLConfig, QwenVL, qwen_tokenizer
from util import get_padding_batch_input, load_image

# set the random seed to ensure reproducibility
random.seed(1337)
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)


# training config settings
MODEL_DIR = "./model/"
VIT_MODEL_FILE = "./model/vit_pretrain.pth"
ADAPTER_MODEL_FILE = "./model/adapter_pretrain.pth"

IMAGE_DIR = "./vlm_data/image_data/"
SFT_JSON_DATA = "./vlm_data/conversation.json"

IMAGE_RESOLUTION = 224



def prepare_prompt_tokens(json_list):
    """
    'image_id': 306,
    'conversations': [
        {
            'topic': 'Objects and their locations',
            'content': [
                {
                    'from': 'user',
                    'value': "What's the main focus in the image?"
                },
                {
                    'from': 'gpt',
                    'value': 'The main focus of the image seems to be a silver car parked at the end of a row in a parking lot.'
                },
                ... 
            ]
        },
    ]
    """
    batch_data = []

    for element in json_list:
        image_id = element["image_id"]
        image_file = f"{IMAGE_DIR}/{image_id}.jpg"

        if not os.path.isfile(image_file):
            # if the image file doesn't exist, skip entirely
            continue
        image_tensor = load_image(image_file, IMAGE_RESOLUTION)

        for conversation in element["conversations"]:
            content = conversation["content"]
            prompt_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
            ]

            for i in range(0, len(content), 2):
                # skip conversations longer than 3 turns, 7 = 1 + 3 * 2 
                if len(prompt_messages) >= 7:
                    break

                user_prompt = content[i]
                model_answer = content[i + 1]
                if user_prompt["from"] != "user" or model_answer["from"] != "gpt":
                    # invalid data, skip
                    break
                if len(user_prompt["value"]) < 1 or len(model_answer["value"]) < 1:
                    break

                prompt_messages.append({"role": "user", "content": user_prompt["value"]})
                prompt_tokens = qwen_tokenizer.apply_chat_template(
                    prompt_messages,
                    tokenize=True,
                    add_generation_prompt=True
                )
                prompt_messages.append({"role": "assistant", "content": model_answer["value"]})
                full_tokens = qwen_tokenizer.apply_chat_template(
                    prompt_messages,
                    tokenize=True,
                    add_generation_prompt=False
                )
                batch_data.append((image_tensor, prompt_tokens, full_tokens))
    
    return batch_data


def parallel_process_sft_data(sft_json_file):
    total_data_list = []

    # load the file
    with open(sft_json_file) as pfile:
        json_list = json.load(pfile)

    chunk_list = [] 
    chunk_size = len(json_list) // 600
    for index in range(0, len(json_list), chunk_size):
        part = json_list[index: index + chunk_size]
        chunk_list.append(part)


    worker_count = os.cpu_count()
    with Pool(processes=worker_count) as pool:
        for batch_data in pool.imap_unordered(prepare_prompt_tokens, chunk_list):
            total_data_list.extend(batch_data)
    
    return total_data_list


def convert_to_tensor(batch_data):

    image_tensors   = [img_tensor for img_tensor, _, _ in batch_data]
    input_tokens    = [full_tokens for _, _, full_tokens in batch_data]
    question_tokens = [prompt_tokens for _, prompt_tokens, _ in batch_data]

    input_ids, attention_masks = get_padding_batch_input(input_tokens)
    labels = input_ids.clone()
    for i in range(input_ids.shape[0]):
        labels[i, :len(question_tokens[i])] = -100
    
    image_batch_tensor = torch.stack(image_tensors, dim=0)

    return input_ids, attention_masks, image_batch_tensor, labels


class DataLoadeFinetune:

    def __init__(self, sft_data_list, validation_size=1000):
        total_data = sft_data_list

        # reserve the same constant part for validation test
        self.val_data = total_data[:validation_size]

        _tmp_input = total_data[validation_size:]
        random.shuffle(_tmp_input)

        # shuffle the remain data and get the train test split
        self.train_data = _tmp_input[len(_tmp_input) // 10:]
        self.test_data = _tmp_input[:len(_tmp_input) // 10]

        #print(f"data length, train_data: {len(self.train_data)}, test_data: {len(self.test_data)}")
        self.train_position = 0
        self.test_position = 0
        self.val_position = 0
    

    def reset(self):
        self.train_position = 0
        self.test_position = 0
        self.val_position = 0


    def training_data_size(self):
        return len(self.train_data)


    def __fetch_data__(self, data_buf, position, batch_size):
        current_position = position

        buf = data_buf[current_position: current_position + batch_size]
        current_position += batch_size

        if current_position > len(data_buf):
            current_position = current_position % len(data_buf)
            buf.extend(data_buf[:current_position])

        input_ids, attention_masks, image_batch_tensor, labels = convert_to_tensor(buf)

        return input_ids, attention_masks, image_batch_tensor, labels, current_position


    def get_train_batch(self, batch_size):
        input_ids, attention_masks, image_batch_tensor, labels, position = self.__fetch_data__(
            self.train_data, self.train_position, batch_size
        )
        self.train_position = position

        return input_ids, attention_masks, image_batch_tensor, labels


    def get_test_batch(self, batch_size):
        input_ids, attention_masks, image_batch_tensor, labels, position = self.__fetch_data__(
            self.test_data, self.test_position, batch_size
        )
        self.test_position = position
        return input_ids, attention_masks, image_batch_tensor, labels
    
    def get_val_batch(self, batch_size):
        input_ids, attention_masks, image_batch_tensor, labels, position = self.__fetch_data__(
            self.val_data, self.val_position, batch_size
        )
        self.val_position = position
        return input_ids, attention_masks, image_batch_tensor, labels



def configure_optimizers(model, weight_decay, learning_rate):
    # start with all of the parameters that require grad
    param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    # do not weight decay bias, layernorm, and other less than 2 dimension weights
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"num decayed tensors: {len(decay_params)}, with {num_decay_params} parameters")
    print(f"num non-decayed tensors: {len(nodecay_params)}, with {num_nodecay_params} parameters")

    fused = True if torch.cuda.is_available() else False
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, fused=fused)
    return optimizer


def instruction_finetune(model, device, dataloader, batch_size, grad_accum_steps, learning_rate, epoch):
    finetune_steps = dataloader.training_data_size() // (batch_size * grad_accum_steps) * epoch
    print(f"Finetuning steps: {finetune_steps}")

    # smaller weight decay as we are doing finetuning
    optimizer = configure_optimizers(model, weight_decay=0.01, learning_rate=learning_rate)

    for step in range(finetune_steps):
        t0 = time.time()
        
        # validation loop
        if step % 200 == 0 or step == finetune_steps - 1:
            model.eval()
            with torch.no_grad():
                test_steps = 10
                test_loss_accum = 0
                for _ in range(test_steps):
                    text, mask, img, labels = dataloader.get_test_batch(batch_size)
                    text, mask, img, labels = \
                        text.to(device), mask.to(device), img.to(device), labels.to(device)
                    logits, loss = model(input_ids=text, attention_masks=mask, img_tensor=img, labels=labels)

                    loss = loss / test_steps
                    test_loss_accum += loss.detach()
                print(f"validation loss: {test_loss_accum.item():.6f}")

        
        # training loop
        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0

        for _ in range(grad_accum_steps):
            text, mask, img, labels = dataloader.get_train_batch(batch_size)
            text, mask, img, labels = \
                text.to(device), mask.to(device), img.to(device), labels.to(device)

            # mixed precision training
            with torch.autocast(device_type=device, dtype=torch.bfloat16):            
                logits, loss = model(input_ids=text, attention_masks=mask, img_tensor=img, labels=labels)
            
                loss = loss / grad_accum_steps
                loss_accum += loss.detach()
                loss.backward()

        # Gradient Clipping
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        t1 = time.time()

        dt = (t1 - t0)
        # the item() function ship the tensor back from gpu to cpu
        print(f"step {step}, loss: {loss.item():.6f}, dt: {dt * 1000:.2f}ms, norm: {norm:.4f}")
    
    return model



def training_loop():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # config model
    model = QwenVL(QwenVLConfig())
    model.init_vision_encoder(VIT_MODEL_FILE)
    model.init_adapter(ADAPTER_MODEL_FILE)       
    model = model.to(device)
    model.adapter = torch.compile(model.adapter)
    model.vision_encoder = torch.compile(model.vision_encoder)

    # config parameter
    batch_size = 32
    grad_accum_steps = 8
    learning_rate = 1e-5
    epoch = 2

    # config dataloader
    total_data_list = parallel_process_sft_data(SFT_JSON_DATA)
    dataloader = DataLoadeFinetune(total_data_list)
    print(f"total training data: {dataloader.training_data_size()}")


    # instrunction finetuning
    model = instruction_finetune(
        model = model, 
        device = device,
        dataloaer = dataloader,
        batch_size = batch_size,
        grad_accum_steps = grad_accum_steps,
        learning_rate = learning_rate, 
        epoch = epoch
    )

    torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"model_finetune.pth"))


if __name__ == "__main__":
    training_loop()