from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

@dataclass
class VisionTransformerConfig:
    patch_size: int = 16 * 16 * 3
    block_size: int = 512
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class SelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()

        assert config.n_embd % config.n_head == 0

        # key, query, value stored in one big matrix
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.GPT_SCALE_INIT = 1

        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=-1)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Flash Attention, not causal attention 
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # final output layer
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.GPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = SelfAttention(config)

        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class VisionTransformer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.cls_embd = nn.Parameter(torch.randn(config.n_embd))

        self.transformer = nn.ModuleDict(
            dict(
                wproj = nn.Linear(config.patch_size, config.n_embd),
                wpe = nn.Embedding(config.block_size, config.n_embd),
                h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f = nn.LayerNorm(config.n_embd),
            )
        )

        # init params
        self.apply(self._init_weights)


    def _init_weights(self, module):

        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'GPT_SCALE_INIT'):
                # scale down by sqrt of the number of layers
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

            # We can try xavier initialization as well
            #torch.nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            #torch.nn.init.xavier_normal_(module.weight)

    def forward(self, patches):

        B, T, P = patches.shape # shape (B, T, P)
        assert T <= self.config.block_size and P == self.config.patch_size

        # load the position as the range tensor, add extra 1 for the class embedding
        pos = torch.arange(0, T + 1, dtype=torch.long, device=patches.device)
        pos_emb = self.transformer.wpe(pos) # shape (T + 1, n_embd)

        # map the image patches into internal representation
        img_emb = self.transformer.wproj(patches) # shape (B, T, n_embd)

        # add the class embedding
        expanded_cls_emb = self.cls_embd.unsqueeze(0).expand(B, 1, -1)
        augmented_emb = torch.cat((expanded_cls_emb, img_emb), dim=1) # shape (B, T + 1, n_embd)

        # patch embeding + position embedding
        x = augmented_emb + pos_emb

        # transformer blocks
        for block in self.transformer.h:
            x = block(x)

        # the final layer norm
        x = self.transformer.ln_f(x)

        # return the full list of embeddings
        # for contrastive training, we only need the first embedding
        # for multi-modal larguage model training, we will need all the embeddings
        return x
    