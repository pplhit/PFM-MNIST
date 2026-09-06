from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

LABEL_NAMES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_label(label: str) -> int:
    label = str(label).strip().lower()
    if label.isdigit() and 0 <= int(label) <= 9:
        return int(label)
    if label in LABEL_NAMES:
        return LABEL_NAMES.index(label)
    raise ValueError(f"Unknown label {label!r}. Use 0-9, all, or {LABEL_NAMES}.")


def wrap_phase(phi: torch.Tensor) -> torch.Tensor:
    return torch.remainder(phi + math.pi, 2 * math.pi) - math.pi


def normalize_sum(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = x.clamp_min(0.0)
    return x / (x.sum(dim=(-2, -1), keepdim=True) + eps)


def normalize_for_display(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    b = x.shape[0]
    flat = x.reshape(b, -1)
    lo = flat.min(dim=1).values.reshape(b, 1, 1, 1)
    hi = flat.max(dim=1).values.reshape(b, 1, 1, 1)
    return (x - lo) / (hi - lo + eps)


def total_variation(x: torch.Tensor) -> torch.Tensor:
    return (x[..., :, 1:] - x[..., :, :-1]).abs().mean() + (
        x[..., 1:, :] - x[..., :-1, :]
    ).abs().mean()


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return x
    channels = x.shape[1]
    radius = max(1, int(3 * sigma))
    coord = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    kernel = torch.exp(-0.5 * (coord / sigma) ** 2)
    kernel = kernel / kernel.sum()
    kx = kernel.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    ky = kernel.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)
    y = F.pad(x, (radius, radius, 0, 0), mode="reflect")
    y = F.conv2d(y, kx, groups=channels)
    y = F.pad(y, (0, 0, radius, radius), mode="reflect")
    return F.conv2d(y, ky, groups=channels)


def make_noisy_seed(clean: torch.Tensor, t_min: float, pure_noise_prob: float) -> torch.Tensor:
    eps = torch.rand_like(clean)
    b = clean.shape[0]
    use_pure = torch.rand(b, 1, 1, 1, device=clean.device) < pure_noise_prob
    t = t_min + (1.0 - t_min) * torch.rand(b, 1, 1, 1, device=clean.device)
    mixed = (1.0 - t) * clean + t * eps
    return torch.where(use_pure, eps, mixed)
