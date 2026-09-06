from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from pfm_mnist.optics import AngularSpectrumPropagator
from pfm_mnist.utils import normalize_sum, wrap_phase


class FrozenLabelTextEncoder(nn.Module):
    """Frozen digit-word embedding as a lightweight text encoder for MNIST."""

    def __init__(self, text_dim: int = 64, seed: int = 123) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        weight = torch.randn(10, text_dim, generator=gen)
        weight = F.normalize(weight, dim=1)
        self.embedding = nn.Embedding(10, text_dim)
        self.embedding.weight.data.copy_(weight)
        self.embedding.weight.requires_grad_(False)

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        return self.embedding(labels)


class FiLM(nn.Module):
    def __init__(self, channels: int, cond_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, channels * 2), nn.SiLU(), nn.Linear(channels * 2, channels * 2)
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.net(cond).chunk(2, dim=1)
        return x * (1.0 + 0.15 * gamma[:, :, None, None]) + 0.15 * beta[:, :, None, None]


class ResBlock(nn.Module):
    def __init__(self, channels: int, cond_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.film = FiLM(channels, cond_dim)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.film(h, cond)
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class ConditionalPhaseUNet(nn.Module):
    """Generate an input-dependent phase stack.

    Output channels are [phi0, psi1, ..., psiL]. Diversity enters through both
    a spatial seed map and a latent vector. This is deliberately not a fixed
    phase mask: different inputs generate different optical velocity fields.
    """

    def __init__(
        self,
        text_dim: int,
        latent_dim: int,
        out_planes: int,
        base_channels: int = 64,
        phase_scale: float = math.pi,
    ) -> None:
        super().__init__()
        self.phase_scale = float(phase_scale)
        cond_dim = base_channels * 2
        self.cond_mlp = nn.Sequential(
            nn.Linear(text_dim + latent_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim)
        )
        self.in_conv = nn.Conv2d(3, base_channels, 3, padding=1)
        self.rb1 = ResBlock(base_channels, cond_dim)
        self.down1 = nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1)
        self.rb2 = ResBlock(base_channels * 2, cond_dim)
        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 4, 4, stride=2, padding=1)
        self.mid1 = ResBlock(base_channels * 4, cond_dim)
        self.mid2 = ResBlock(base_channels * 4, cond_dim)
        self.up1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1)
        self.rb3 = ResBlock(base_channels * 2, cond_dim)
        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1)
        self.rb4 = ResBlock(base_channels, cond_dim)
        self.out = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_planes, 3, padding=1),
        )

    @staticmethod
    def make_coord(batch: int, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        yy = torch.linspace(-1, 1, h, device=device, dtype=dtype)
        xx = torch.linspace(-1, 1, w, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
        return torch.stack([grid_x, grid_y], dim=0)[None].repeat(batch, 1, 1, 1)

    def forward(self, seed_map: torch.Tensor, text_emb: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        b, _, h, w = seed_map.shape
        cond = self.cond_mlp(torch.cat([text_emb, latent], dim=1))
        coord = self.make_coord(b, h, w, seed_map.device, seed_map.dtype)
        h1 = self.rb1(self.in_conv(torch.cat([seed_map, coord], dim=1)), cond)
        h2 = self.rb2(self.down1(h1), cond)
        h3 = self.mid2(self.mid1(self.down2(h2), cond), cond)
        h = self.rb3(self.up1(h3) + h2, cond)
        h = self.rb4(self.up2(h) + h1, cond)
        return wrap_phase(self.phase_scale * torch.tanh(self.out(h)))


class PFMGenerator(nn.Module):
    """PFM generator: conditional phases + differentiable optical propagation."""

    def __init__(
        self,
        size: int = 64,
        text_dim: int = 64,
        latent_dim: int = 64,
        num_layers: int = 4,
        base_channels: int = 64,
        wavelength: float = 635e-9,
        dx: float = 3.6e-6,
        dz: float = 0.04,
        pad_factor: int = 2,
        source_floor: float = 0.03,
        phase_scale: float = math.pi,
    ) -> None:
        super().__init__()
        self.size = int(size)
        self.latent_dim = int(latent_dim)
        self.num_layers = int(num_layers)
        self.wavelength = float(wavelength)
        self.dx = float(dx)
        self.dz = float(dz)
        self.k = 2 * math.pi / self.wavelength
        self.source_floor = float(source_floor)
        self.text_encoder = FrozenLabelTextEncoder(text_dim=text_dim)
        self.phase_generator = ConditionalPhaseUNet(
            text_dim=text_dim,
            latent_dim=latent_dim,
            out_planes=num_layers + 1,
            base_channels=base_channels,
            phase_scale=phase_scale,
        )
        self.propagator = AngularSpectrumPropagator(wavelength, dx, dz, pad_factor)

    def sample_latent(self, batch: int, device: torch.device) -> torch.Tensor:
        return torch.randn(batch, self.latent_dim, device=device)

    def make_input_density(self, seed_map: torch.Tensor) -> torch.Tensor:
        return normalize_sum(seed_map.clamp_min(0.0) + self.source_floor)

    def forward(
        self,
        seed_map: torch.Tensor,
        labels: torch.Tensor,
        latent: torch.Tensor | None = None,
        return_fields: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        if latent is None:
            latent = self.sample_latent(seed_map.shape[0], seed_map.device)
        text_emb = self.text_encoder(labels)
        phase_maps = self.phase_generator(seed_map, text_emb, latent)
        rho0 = self.make_input_density(seed_map)
        field = torch.sqrt(rho0 + 1e-12) * torch.exp(1j * phase_maps[:, 0:1])
        intensities = [rho0]
        velocity_phases = []
        for layer in range(self.num_layers):
            field = field * torch.exp(1j * phase_maps[:, layer + 1 : layer + 2])
            velocity_phases.append(torch.angle(field))
            field = self.propagator(field)
            intensities.append(normalize_sum(field.abs() ** 2))
        out = normalize_sum(field.abs() ** 2)
        if not return_fields:
            return out
        return out, {
            "rho0": rho0,
            "phase_maps": phase_maps,
            "velocity_phases": velocity_phases,
            "intensities": intensities,
            "field": field,
            "latent": latent,
        }


class MinibatchStdDev(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        std = torch.sqrt(x.var(dim=0, unbiased=False) + 1e-8).mean().view(1, 1, 1, 1)
        return torch.cat([x, std.expand(x.shape[0], 1, x.shape[2], x.shape[3])], dim=1)


class ConditionalDiscriminator(nn.Module):
    """Small projection discriminator for conditional MNIST generation."""

    def __init__(self, size: int = 64, embed_dim: int = 128, base_channels: int = 48) -> None:
        super().__init__()
        self.label_emb = nn.Embedding(10, embed_dim)
        self.conv1 = nn.Conv2d(1, base_channels, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(base_channels * 2, base_channels * 4, 4, stride=2, padding=1)
        self.stddev = MinibatchStdDev()
        self.conv4 = nn.Conv2d(base_channels * 4 + 1, embed_dim, 4, stride=2, padding=1)
        final_spatial = size // 16
        self.head = nn.Linear(embed_dim * final_spatial * final_spatial, 1)
        self.proj = nn.Linear(embed_dim * final_spatial * final_spatial, embed_dim)

    def forward(self, image: torch.Tensor, labels: torch.Tensor, return_features: bool = False):
        features = []
        h = F.leaky_relu(self.conv1(image), 0.2)
        features.append(h)
        h = F.leaky_relu(self.conv2(h), 0.2)
        features.append(h)
        h = F.leaky_relu(self.conv3(h), 0.2)
        features.append(h)
        h = self.stddev(h)
        h = F.leaky_relu(self.conv4(h), 0.2)
        features.append(h)
        flat = h.flatten(1)
        logit = self.head(flat)
        proj = (self.proj(flat) * self.label_emb(labels)).sum(dim=1, keepdim=True)
        logit = logit + proj / math.sqrt(self.label_emb.embedding_dim)
        if return_features:
            return logit, features
        return logit
