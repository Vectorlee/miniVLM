from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from vision_transformer import VisionTransformer, VisionTransformerConfig
from text_encoder import TextEncoder, TextEncoderConfig


def clip_loss(text_embds, vision_embds, scaler):
    B, C = text_embds.shape

    logits = vision_embds @ text_embds.transpose(0, 1) * scaler

    labels = torch.arange(B, device=text_embds.device)
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.transpose(0, 1), labels)

    loss = (loss_i + loss_t) / 2
    return loss


class CLIPModel(nn.Module):

    def __init__(self, text_config: TextEncoderConfig, vision_config: VisionTransformerConfig, hidden_dim=512):
        super().__init__()

        self.text_encoder = TextEncoder(text_config)
        self.vision_encoder = VisionTransformer(vision_config)

        self.text_proj = nn.Linear(text_config.n_embd, hidden_dim)
        self.vision_proj = nn.Linear(vision_config.n_embd, hidden_dim)

        # initialize the temperature as 0.07
        self.temperature = nn.Parameter(torch.tensor([0.07], dtype=torch.float32))


    def forward(self, input_ids, attention_masks, img_tensor):
        B1, T1 = input_ids.shape
        B2, C, H, W = img_tensor.shape
        assert B1 == B2

        text_embds = self.text_encoder(input_ids) # [B1, T1, C1]
        img_embds = self.vision_encoder(img_tensor)  # [B2, T2 + 1, C2]

        indexes = attention_masks.sum(dim=1) - 1 # [B1,]
        # the last token embedding
        text_encodings = text_embds[torch.arange(B1, device=text_embds.device), indexes] # [B1, C1]
        # the first class embedding
        vision_encodings = img_embds[torch.arange(B2, device=img_embds.device), 0] # [B2, C2]

        # [B, hidden_dim]
        text_projs = F.normalize(self.text_proj(text_encodings), p=2.0, dim=1)
        vision_projs = F.normalize(self.vision_proj(vision_encodings), p=2.0, dim=1)

        # make sure the scaler does not exceed 100
        scaler = torch.clamp(torch.exp(self.temperature), max=100)
        
        # logits = vision_projs @ text_projs.transpose(0, 1) * scaler
        # labels = torch.arange(B1, device=img_tensor.device)
        # loss_i = F.cross_entropy(logits, labels)
        # loss_t = F.cross_entropy(logits.transpose(0, 1), labels)
        # loss = (loss_i + loss_t) / 2

        return text_projs, vision_projs, scaler




