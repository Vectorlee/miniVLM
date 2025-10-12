from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from vision_transformer import VisionTransformer, VisionTransformerConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from util import strip_state_prefix

LLM_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

qwen_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
im_start = qwen_tokenizer("<|im_start|>").input_ids[0]
im_end = qwen_tokenizer("<|im_end|>").input_ids[0]


@dataclass
class QwenVLConfig:
    query_size: int = 196
    vision_embd: int = 768
    text_embd: int = 2048   # qwen2.5-3b
    n_head: int = 8

class VLAdaptor(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.query = nn.Parameter(torch.randn(config.query_size, config.vision_embd))

        self.key_proj = nn.Linear(config.vision_embd, config.vision_embd)
        self.value_proj = nn.Linear(config.vision_embd, config.text_embd)
        self.output_proj = nn.Linear(config.text_embd, config.text_embd)

    def forward(self, vision_features):
        B, T, C = vision_features.shape
        Q, d_q = self.query.shape
        assert C == self.config.vision_embd and Q == self.config.query_size

        # [n_head, Q, d_q // n_head]
        q = self.query.view(Q, self.config.n_head, -1).transpose(0, 1)

        # [B, T, d_k], d_k = d_q = C = vision_embd
        k = self.key_proj(vision_features)
        # [B, n_head, T, d_k // n_head]
        k = k.view(B, T, self.config.n_head, -1).transpose(1, 2)

        # [B, T, d_v], d_v = text_embed
        v = self.value_proj(vision_features)
        # [B, n_head, T, d_v // head]
        v = v.view(B, T, self.config.n_head, -1).transpose(1, 2)
        
        # query: [_, n_head, Q, d_q //n_head], kT: [B, n_head, d_k // n_head, T], broadcasting here
        # cross attention: [B, n_head, Q, T]
        cross_attention = q @ k.transpose(2, 3) / math.sqrt(C) # d_k = C
        cross_attention = F.softmax(cross_attention, dim=-1)

        # [B, n_head, Q, T] @ [B, n_head, T, d_v // n_head] -> [B, n_head, Q, d_v // n_head]
        output = cross_attention @ v
        # [B, Q, d_v]
        output = output.transpose(1, 2).contiguous().view(B, Q, -1)

        return self.output_proj(output)


class QwenVL(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.vision_encoder = VisionTransformer(VisionTransformerConfig())
        self.llm_backbone = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME)
        self.adapter = VLAdaptor(config)

    
    def init_vision_encoder_from_clip(self, clip_model_file):
        vit_prefix = "vision_encoder."

        with torch.no_grad():
            state_dict = strip_state_prefix(torch.load(clip_model_file))
            sub_dict = {k.replace(vit_prefix, ""): v for k, v in state_dict.items() if k.startswith(vit_prefix)}
            self.vision_encoder.load_state_dict(sub_dict)
        return

    def init_vision_encoder(self, vit_model_file):

        with torch.no_grad():
            state_dict = strip_state_prefix(torch.load(vit_model_file))
            self.vision_encoder.load_state_dict(state_dict)
        return
    
    def init_adapter(self, adapter_model_file):

        with torch.no_grad():
            state_dict = strip_state_prefix(torch.load(adapter_model_file))
            self.adapter.load_state_dict(state_dict)
        return

    def freeze_llm_backbone(self):
        for param in self.llm_backbone.parameters():
            param.requires_grad = False
        return 

    def freeze_vision_encoder(self):
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        return


    def forward(self, input_ids, attention_masks, img_tensor, labels=None):
        B, T = input_ids.shape

        # text imbeddings [B, T, C]
        text_embds = self.llm_backbone.model.embed_tokens(input_ids)

        # expand the bracket tokens to match batch size [B, 1, C]
        im_start_embds = self.llm_backbone.model.embed_tokens(torch.tensor([im_start], device=img_tensor.device)).expand(B, 1, -1)
        im_end_embds = self.llm_backbone.model.embed_tokens(torch.tensor([im_end], device=img_tensor.device)).expand(B, 1, -1)

        # vision embeddings [B, K + 1, P]
        vision_embds = self.vision_encoder(img_tensor)

        # remove the first cls embedding
        _, K, _ = vision_embds.shape
        vision_embds = vision_embds[torch.arange(B, device=img_tensor.device), 1:K] # [B, K, P]

        # project to llm embedding space [B, Q, C]
        proj_embds = self.adapter(vision_embds)
        _, Q, _ = proj_embds.shape
        
        # add the image embeddings to the front [B, 2 + Q + T, C]
        # <|im_start|>img embeddings<|im_end|>
        input_embds = torch.cat((im_start_embds, proj_embds, im_end_embds, text_embds), dim=1)

        # add image masks
        img_mask = torch.ones(B, Q + 2, dtype=attention_masks.dtype, device=attention_masks.device)
        input_masks = torch.cat((img_mask, attention_masks), dim=1)

        # add extra negative labels
        output_labels = labels
        if labels:
            extra_labels = torch.zeros(B, Q + 2, dtype=labels.dtype, device=labels.device).fill_(-100)
            output_labels = torch.cat((extra_labels, labels), dim=1)

        output = self.llm_backbone(inputs_embeds=input_embds, attention_mask=input_masks, labels=output_labels)
        return output.logits, output.loss
