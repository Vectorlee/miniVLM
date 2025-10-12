from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
import torch
import random
import argparse

from qwenvl_model import QwenVL, QwenVLConfig, qwen_tokenizer
from util import load_image


random.seed(1337)
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

im_start = qwen_tokenizer("<|im_start|>").input_ids[0]
im_end = qwen_tokenizer("<|im_end|>").input_ids[0]



def generate(model, input_ids, attention_masks, img_tensor, temperature, max_steps):
    B, T = input_ids.shape
    finish_index = torch.zeros(B, dtype=torch.int64)

    img_tokens_len = model.config.query_size + 2
    for _ in range(max_steps):
        logits, _ = model(input_ids, attention_masks, img_tensor)   # [B, T, vocab_size]

        # there are <im_start> [img_query] <im_end> in the front
        last_logit_indexes = attention_masks.sum(dim=1) - 1
        last_logit = logits[torch.arange(B, device=input_ids.device), last_logit_indexes + img_tokens_len]

        # expend the length of input tensors
        pad_tensor = torch.zeros(B, 1, dtype=torch.int64, device=input_ids.device)
        input_ids = torch.cat((input_ids, pad_tensor), dim = 1)
        attention_masks = torch.cat((attention_masks, pad_tensor), dim = 1)

        next_tokens = torch.tensor([])
        if temperature == 0:
            # just pick the most likely token
            next_tokens = torch.argmax(last_logit, dim=1)
        else:
            # apply the temperature then do softmax
            probs = F.softmax(last_logit / temperature, dim=1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze()
        
        # assign the next token and attention mask
        input_ids[torch.arange(B, device=input_ids.device), last_logit_indexes] = next_tokens
        attention_masks[torch.arange(B, device=input_ids.device), last_logit_indexes] = 1

        # if we hit the endoftext token, mark it in the finish_mask 
        idxs = torch.nonzero(next_tokens == im_end, as_tuple=False).squeeze().tolist()
        for i in idxs:
            finish_index[i] = finish_index[i] if finish_index[i] != 0 else last_logit_indexes[i].item() + 1

        if torch.nonzero(finish_index).shape[0] == B:
            break

    return input_ids, attention_masks, finish_index


def decode_generation(input_ids, attention_masks, finish_index):
    B, T = input_ids.shape
    indexes = attention_masks.sum(dim=1).tolist()
    
    answer_list = []
    for i in range(B):
        sequence = input_ids[i].tolist()
        if finish_index[i] > 0:
            sequence = sequence[:finish_index[i]]
        else:
            sequence = sequence[:indexes[i]]
        answer_list.append(qwen_tokenizer.decode(sequence))

    return answer_list


def vision_language_generation(model, img_file, prompt, temperature=0.8):
    device = next(model.parameters()).device

    img_tensor = load_image(img_file, resolution=224).unsqueeze(0)

    prompt_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    prompt_tokens = qwen_tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True
    )

    # [1, token_len]
    input_ids = torch.tensor(prompt_tokens).unsqueeze(0)
    attention_mask = torch.ones(len(prompt_tokens), dtype=torch.int64).unsqueeze(0)
    input_ids, attention_mask, img_tensor = input_ids.to(device), attention_mask.to(device), img_tensor.to(device)
    
    with torch.no_grad():
        answer, mask, finish_index = generate(
            model = model, 
            input_ids = input_ids,
            attention_masks = attention_mask,
            img_tensor = img_tensor, 
            temperature = temperature,
            max_steps = 100
        )

    output = decode_generation(answer, mask, finish_index)
    return output


def main():
    parser = argparse.ArgumentParser(description="Example script that takes two string arguments.")
    parser.add_argument("--model", type=str, required=True, help="The saved model pth file")
    parser.add_argument("--image", type=str, required=True, help="The image file")
    parser.add_argument("--prompt", type=str, required=True, help="The text prompt")
    
    args = parser.parse_args()

    # TODO: add the code



if __name__ == "__main__":
    VIT_MODEL_FILE = "./clip_data/vit_pretrain.pth"
    ADAPTER_MODEL_FILE = "./clip_data/adapter_pretrain.pth"
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = QwenVL(QwenVLConfig())

    model.init_vision_encoder(VIT_MODEL_FILE)
    model.init_adapter(ADAPTER_MODEL_FILE)       
    model = model.to(device)
    model.eval()

    input_prompt = "Please describe the content of this image"
    input_image = "./clip_data/test_imge.jpg"
    
    output = vision_language_generation(model, input_image, input_prompt)
    print(output)