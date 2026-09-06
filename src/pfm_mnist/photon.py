from __future__ import annotations

import torch
import torch.nn.functional as F

from pfm_mnist.models import PFMGenerator
from pfm_mnist.optics import phase_gradient
from pfm_mnist.utils import normalize_sum


@torch.no_grad()
def sample_initial_photons(rho0: torch.Tensor, n_photons: int, dx: float) -> torch.Tensor:
    if rho0.shape[0] != 1:
        raise ValueError("Photon decoding is implemented for one generated image at a time.")
    _, _, h, w = rho0.shape
    prob = normalize_sum(rho0)[0, 0].reshape(-1)
    ids = torch.multinomial(prob, n_photons, replacement=True)
    yy = torch.div(ids, w, rounding_mode="floor")
    xx = ids % w
    x = (xx.float() + torch.rand_like(xx.float()) - w / 2.0) * dx
    y = (yy.float() + torch.rand_like(yy.float()) - h / 2.0) * dx
    return torch.stack([x, y], dim=-1)


@torch.no_grad()
def sample_grid(values: torch.Tensor, xy_m: torch.Tensor, dx: float) -> torch.Tensor:
    if values.shape[0] != 1:
        raise ValueError("Grid sampling expects one field.")
    _, _, h, w = values.shape
    x_norm = xy_m[:, 0] / ((w - 1) * dx / 2.0)
    y_norm = xy_m[:, 1] / ((h - 1) * dx / 2.0)
    grid = torch.stack([x_norm, y_norm], dim=-1).view(1, -1, 1, 2).clamp(-1, 1)
    sampled = F.grid_sample(values, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return sampled[0, :, :, 0].T


@torch.no_grad()
def integrate_photon_flow(model: PFMGenerator, info: dict, n_photons: int = 100000) -> torch.Tensor:
    """Discrete characteristic integration: X_{l+1}=X_l+dz*grad(phi_l(X_l))/k."""
    rho0 = info["rho0"][:1]
    xy = sample_initial_photons(rho0, n_photons, model.dx).to(rho0.device)
    half = (model.size - 1) * model.dx / 2.0
    for phase in info["velocity_phases"]:
        gx, gy = phase_gradient(phase[:1], model.dx)
        velocity = sample_grid(torch.cat([gx, gy], dim=1), xy, model.dx) / model.k
        xy = xy + model.dz * velocity
        xy[:, 0].clamp_(-half, half)
        xy[:, 1].clamp_(-half, half)
    return xy


@torch.no_grad()
def photon_histogram(xy: torch.Tensor, size: int, dx: float, device: torch.device) -> torch.Tensor:
    h = w = size
    x_pix = torch.floor(xy[:, 0] / dx + w / 2.0).long()
    y_pix = torch.floor(xy[:, 1] / dx + h / 2.0).long()
    valid = (x_pix >= 0) & (x_pix < w) & (y_pix >= 0) & (y_pix < h)
    ids = y_pix[valid] * w + x_pix[valid]
    hist = torch.zeros(h * w, device=device)
    hist.scatter_add_(0, ids, torch.ones_like(ids, dtype=hist.dtype))
    return normalize_sum(hist.view(1, 1, h, w))
