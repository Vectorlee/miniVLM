from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
import torch
import random
import argparse
import os

from qwenvl_model import QwenVL, QwenVLConfig, qwen_tokenizer
from util import load_image, strip_state_prefix


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
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
        
        # assign the next token and attention mask
        input_ids[torch.arange(B, device=input_ids.device), last_logit_indexes + 1] = next_tokens
        attention_masks[torch.arange(B, device=input_ids.device), last_logit_indexes + 1] = 1

        # if we hit the endoftext token, mark it in the finish_mask 
        idxs = torch.nonzero(next_tokens == im_end, as_tuple=False).squeeze(-1).tolist()
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


def decode_one_response(respose_token, prompt_token):
    answer_token = respose_token[len(prompt_token):]
    answer = qwen_tokenizer.decode(answer_token).strip()

    answer = answer[len("<|im_start|>"): ] if answer.startswith("<|im_start|>") else answer
    answer = answer[: -len("<|im_end|>")] if answer.endswith("<|im_end|>") else answer
    return answer


def vision_language_generation(model, img_file, prompt_tokens, temperature=0.8):
    device = next(model.parameters()).device
    
    img_tensor = load_image(img_file, resolution=224).unsqueeze(0)

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
    
    output = answer[0]
    if finish_index[0] > 0:
        output = output[:finish_index[0]]

    return output


def main():
    parser = argparse.ArgumentParser(description="Example script that takes two string arguments.")
    parser.add_argument("--model", type=str, required=True, help="The saved model pth file")
    parser.add_argument("--image", type=str, required=True, help="The image file")

    args = parser.parse_args()
    model_file = args.model
    image_file = args.image

    if not os.path.isfile(model_file):
        print(f"Model file {model_file} does not exist")
        return
    if not os.path.isfile(image_file):
        print(f"Image file {image_file} does not exist")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = QwenVL(QwenVLConfig())
    model.load_state_dict(strip_state_prefix(torch.load(model_file, weights_only=True)))
    model = model.to(device)

    prompt1 = input("User Prompt: ")

    prompt_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt1}
    ]
    prompt_tokens1 = qwen_tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True
    )
    response_tokens1 = vision_language_generation(model, image_file, prompt_tokens1)
    answer1 = decode_one_response(response_tokens1, prompt_tokens1)
    print(f"Assistant: {answer1}")

    prompt2 = input("User Prompt: ")
    prompt_messages.append({"role": "assistant", "content": answer1})
    prompt_messages.append({"role": "user", "content": prompt2})
    prompt_tokens2 = qwen_tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True
    )
    response_tokens2 = vision_language_generation(model, image_file, prompt_tokens2)
    answer2 = decode_one_response(response_tokens2, prompt_tokens2)
    print(f"Assistant: {answer2}")

    return



if __name__ == "__main__":
    main()