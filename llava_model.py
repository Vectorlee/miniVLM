from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
from vision_transformer import VisionTransformer, VisionTransformerConfig
from transformers import AutoModelForCausalLM

LLM_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

class LLaVA(nn.Module):

    def __init__(self):
        super().__init__()

        self.vision_encoder = VisionTransformer(VisionTransformerConfig())
        self.llm_backbone = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME)
        self.adapter = nn.Linear(self.vision_encoder.config.n_embd, self.llm_backbone.config.hidden_size)

    def freeze_llm_backbone(self):
        for param in self.llm_backbone.parameters():
            param.requires_grad = False
        return 

    def freeze_vision_encoder(self):
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        return
    
    def forward(self, input_ids, attention_masks, img_tensor, labels=None):
        # text imbeddings [B, T, C]
        text_embds = self.llm_backbone.model.embed_tokens(input_ids)

        # vision embeddings [B, K + 1, P]
        vision_embds = self.vision_encoder(img_tensor)

        # remove the first cls embedding
        B, K, P = vision_embds.shape
        vision_embds = vision_embds[torch.arange(B, device=img_tensor.device), 1:K] # [B, K, P]

        # project to llm embedding space [B, K, C]
        proj_embds = self.adapter(vision_embds)
        
        # add the image embeddings to the front: [B, K + T, C]
        input_embds = torch.cat((proj_embds, text_embds), dim=1)

        # add image masks
        img_mask = torch.ones(B, K - 1, dtype=attention_masks.dtype, device=attention_masks.device)
        input_masks = torch.cat((img_mask, attention_masks), dim=1)

        output = self.llm_backbone(inputs_embeds=input_embds, attention_mask=input_masks, labels=labels)
        return output.logits, output.loss
