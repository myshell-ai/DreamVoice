import torch
import torch.nn as nn
from diffusers import UNet2DModel, UNet2DConditionModel
import yaml
from einops import repeat, rearrange

from typing import Any
from torch import Tensor


def rand_bool(shape: Any, proba: float, device: Any = None) -> Tensor:
    if proba == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif proba == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.bernoulli(torch.full(shape, proba, device=device)).to(torch.bool)


class FixedEmbedding(nn.Module):
    def __init__(self, features=128):
        super().__init__()
        self.embedding = nn.Embedding(1, features)

    def forward(self, y):
        B, L, C, device = y.shape[0], y.shape[-2], y.shape[-1], y.device
        embed = self.embedding(torch.zeros(B, device=device).long())
        fixed_embedding = repeat(embed, "b c -> b l c", l=L)
        return fixed_embedding


class P2E_Cross(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.unet = UNet2DConditionModel(**self.config['unet'])
        self.unet.set_use_memory_efficient_attention_xformers(True)
        self.cfg_embedding = FixedEmbedding(self.config['unet']['cross_attention_dim'])

        self.context_embedding = nn.Sequential(
            nn.Linear(self.config['unet']['cross_attention_dim'], self.config['unet']['cross_attention_dim']),
            nn.SiLU(),
            nn.Linear(self.config['unet']['cross_attention_dim'], self.config['unet']['cross_attention_dim']))

    def forward(self, target, t, prompt, prompt_mask=None,
                train_cfg=False, cfg_prob=0.0):
        B, C = target.shape
        target = target.unsqueeze(-1).unsqueeze(-1)

        if train_cfg:
            if cfg_prob > 0.0:
                # Randomly mask embedding
                batch_mask = rand_bool(shape=(B, 1, 1), proba=cfg_prob, device=target.device)
                fixed_embedding = self.cfg_embedding(prompt).to(target.dtype)
                prompt = torch.where(batch_mask, fixed_embedding, prompt)

        prompt = self.context_embedding(prompt)
        # fix the bug that prompt will copy dtype from target in diffusers
        target = target.to(prompt.dtype)

        output = self.unet(sample=target, timestep=t,
                           encoder_hidden_states=prompt,
                           encoder_attention_mask=prompt_mask)['sample']

        return output.squeeze(-1).squeeze(-1)


if __name__ == "__main__":
    with open('p2e_cross.yaml', 'r') as fp:
        config = yaml.safe_load(fp)
    device = 'cuda'

    model = P2E_Cross(config['diffwrap']).to(device)

    x = torch.rand((2, 256)).to(device)
    t = torch.randint(0, 1000, (2,)).long().to(device)
    prompt = torch.rand(2, 64, 768).to(device)
    prompt_mask = torch.ones(2, 64).to(device)

    output = model(x, t, prompt, prompt_mask, train_cfg=True, cfg_prob=0.25)