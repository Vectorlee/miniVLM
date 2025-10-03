from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from vision_transformer import VisionTransformer, VisionTransformerConfig
from text_encoder import TextEncoder, TextEncoderConfig


@dataclass
class CLIPConfig:
    text_config: TextEncoderConfig = TextEncoderConfig()
    vision_config: VisionTransformerConfig = VisionTransformerConfig()
    hidden_dim: int = 512


class CLIPModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.text_encoder = TextEncoder(config.text_config)
        self.vision_encoder = VisionTransformer(config.vision_config)

        self.text_proj = nn.Linear(config.text_config.n_embd, config.hidden_dim)
        self.vision_proj = nn.Linear(config.vision_config.n_embd, config.hidden_dim)

        # initialize the temperature as 0.07
        self.temperature = nn.Parameter(torch.tensor([0.07], dtype=torch.float32))


    def forward(self, input_ids, attention_masks, patches):
        B1, T1 = input_ids.shape
        B2, T2, C2 = patches.shape
        assert B1 == B2

        text_embds = self.text_encoder(input_ids) # [B1, T1, C1]
        img_embds = self.vision_encoder(patches)  # [B2, T2 + 1, C2]

        indexes = attention_masks.sum(dim=1) # [B1,]
        # the last token embedding
        text_encodings = text_embds[torch.arange(B1, device=input_ids.device), indexes] # [B1, C1]
        # the first class embedding
        vision_encodings = img_embds[torch.arange(B2, device=patches.device), 0] # [B2, C2]

        text_projs = F.normalize(self.text_proj(text_encodings), p=2.0, dim=1)
        vision_projs = F.normalize(self.vision_proj(vision_encodings), p=2.0, dim=1)

        # make sure the scaler does not exceed 100
        scaler = torch.clamp(torch.exp(self.temperature), max=100)
        # [B1, hidden_dim] @ [hidden_dim, B2]
        logits = vision_projs @ text_projs.transpose(0, 1) * scaler

        labels = torch.arange(B1, device=patches.device)
        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.transpose(0, 1), labels)

        loss = (loss_i + loss_t) / 2
        return logits, loss




