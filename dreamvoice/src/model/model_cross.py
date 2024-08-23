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


class DiffVC_Cross(nn.Module):
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

        self.content_embedding = nn.Sequential(
            nn.Linear(self.config['cls_embedding']['content_dim'], self.config['cls_embedding']['content_hidden']),
            nn.SiLU(),
            nn.Linear(self.config['cls_embedding']['content_hidden'], self.config['cls_embedding']['content_hidden']))

        if self.config['cls_embedding']['use_pitch']:
            self.pitch_control = True
            self.pitch_embedding = nn.Sequential(
                nn.Linear(self.config['cls_embedding']['pitch_dim'], self.config['cls_embedding']['pitch_hidden']),
                nn.SiLU(),
                nn.Linear(self.config['cls_embedding']['pitch_hidden'],
                          self.config['cls_embedding']['pitch_hidden']))

            self.pitch_uncond = nn.Parameter(torch.randn(self.config['cls_embedding']['pitch_hidden']) /
                                             self.config['cls_embedding']['pitch_hidden'] ** 0.5)
        else:
            print('no pitch module')
            self.pitch_control = False

    def forward(self, target, t, content, prompt, prompt_mask=None, pitch=None,
                train_cfg=False, speaker_cfg=0.0, pitch_cfg=0.0):
        B, C, M, L = target.shape
        content = self.content_embedding(content)
        content = repeat(content, "b t c-> b c m t", m=M)
        target = target.to(content.dtype)
        x = torch.cat([target, content], dim=1)

        if self.pitch_control:
            if pitch is not None:
                pitch = self.pitch_embedding(pitch)
            else:
                pitch = repeat(self.pitch_uncond, "c-> b t c", b=B, t=L).to(target.dtype)

        if train_cfg:
            # Randomly mask embedding
            batch_mask = rand_bool(shape=(B, 1, 1), proba=speaker_cfg, device=target.device)
            fixed_embedding = self.cfg_embedding(prompt).to(target.dtype)
            prompt = torch.where(batch_mask, fixed_embedding, prompt)

            if self.pitch_control:
                batch_mask = rand_bool(shape=(B, 1, 1), proba=pitch_cfg, device=target.device)
                pitch_uncond = repeat(self.pitch_uncond, "c-> b t c", b=B, t=L).to(target.dtype)
                pitch = torch.where(batch_mask, pitch_uncond, pitch)

        prompt = self.context_embedding(prompt)

        if self.pitch_control:
            pitch = repeat(pitch, "b t c-> b c m t", m=M)
            x = torch.cat([x, pitch], dim=1)

        output = self.unet(sample=x, timestep=t,
                           encoder_hidden_states=prompt,
                           encoder_attention_mask=prompt_mask)['sample']

        return output


if __name__ == "__main__":
    with open('diffvc_cross_pitch.yaml', 'r') as fp:
        config = yaml.safe_load(fp)
    device = 'cuda'

    model = DiffVC_Cross(config['diffwrap']).to(device)

    x = torch.rand((2, 1, 100, 256)).to(device)
    y = torch.rand((2, 256, 768)).to(device)
    t = torch.randint(0, 1000, (2,)).long().to(device)
    prompt = torch.rand(2, 64, 768).to(device)
    prompt_mask = torch.ones(2, 64).to(device)
    p = torch.rand(2, 256, 1).to(device)

    output = model(x, t, y, prompt, prompt_mask, p, train_cfg=True, speaker_cfg=0.25, pitch_cfg=0.5)