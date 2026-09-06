from __future__ import annotations

import torch
import torch.nn.functional as F

from pfm_mnist.optics import phase_gradient
from pfm_mnist.utils import gaussian_blur, normalize_for_display, normalize_sum, total_variation


def prepare_density_target(
    target: torch.Tensor,
    target_floor: float = 1e-4,
    blur_sigma: float = 0.6,
) -> torch.Tensor:
    target = gaussian_blur(target, blur_sigma)
    target = target + target_floor
    return normalize_sum(target)


def endpoint_density_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    target_floor: float = 1e-4,
    blur_sigma: float = 0.6,
) -> torch.Tensor:
    """Distributional endpoint loss for normalized photon density.

    It uses Hellinger distance plus weak symmetric KL to compare spatial
    probability densities.
    """
    eps = 1e-8
    pred = normalize_sum(pred + eps)
    target = prepare_density_target(target, target_floor, blur_sigma)

    hellinger = ((torch.sqrt(pred + eps) - torch.sqrt(target + eps)) ** 2).sum(dim=(-2, -1)).mean()
    kl_t2p = (target * (torch.log(target + eps) - torch.log(pred + eps))).sum(dim=(-2, -1)).mean()
    kl_p2t = (pred * (torch.log(pred + eps) - torch.log(target + eps))).sum(dim=(-2, -1)).mean()
    return hellinger + 0.05 * kl_t2p + 0.01 * kl_p2t


def multiscale_density_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    target_floor: float = 1e-4,
    blur_sigma: float = 0.6,
) -> torch.Tensor:
    """Multi-scale image-domain density loss.

    This stabilizes the global digit shape and helps avoid single bright spots.
    """
    pred = normalize_for_display(normalize_sum(pred))
    target = normalize_for_display(prepare_density_target(target, target_floor, blur_sigma))
    loss = 0.0
    for sigma, weight in [(0.0, 1.0), (1.0, 0.7), (2.0, 0.5), (4.0, 0.3)]:
        loss = loss + weight * F.l1_loss(gaussian_blur(pred, sigma), gaussian_blur(target, sigma))
    return loss


def anti_collapse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lambda_ipr: float,
    lambda_peak: float,
    target_floor: float = 1e-4,
    blur_sigma: float = 0.6,
) -> torch.Tensor:
    """Suppress local photon-density collapse."""
    eps = 1e-8
    pred = normalize_sum(pred + eps)
    target = prepare_density_target(target, target_floor, blur_sigma)

    ipr_pred = (pred**2).sum(dim=(-2, -1))
    ipr_target = (target**2).sum(dim=(-2, -1)).detach()
    ipr = ((ipr_pred / (ipr_target + eps) - 1.0) ** 2).mean()

    peak_ratio = pred.amax(dim=(-2, -1)) / (target.amax(dim=(-2, -1)).detach() + eps)
    peak = F.relu(peak_ratio - 1.8).pow(2).mean()
    return lambda_ipr * ipr + lambda_peak * peak


def diversity_loss(
    out_a: torch.Tensor,
    out_b: torch.Tensor,
    latent_a: torch.Tensor,
    latent_b: torch.Tensor,
    margin: float = 0.08,
) -> torch.Tensor:
    """Latent-sensitive diversity regularizer.

    Same label with different latent/noise seeds should produce measurably
    different photon-density patterns. This encourages class-conditional sample
    diversity through different phase-induced transport maps.
    """
    a = gaussian_blur(normalize_for_display(out_a), sigma=1.2)
    b = gaussian_blur(normalize_for_display(out_b), sigma=1.2)
    image_distance = (a - b).abs().flatten(1).mean(dim=1)
    latent_distance = (latent_a - latent_b).abs().flatten(1).mean(dim=1).detach()
    normalized_distance = image_distance / (latent_distance + 1e-6)
    return F.relu(margin - normalized_distance).mean()


def phase_smoothness_loss(phase_maps: torch.Tensor) -> torch.Tensor:
    return total_variation(phase_maps)


def _sample_points_from_density(density: torch.Tensor, num_points: int) -> torch.Tensor:
    """Sample normalized coordinates from a batch of spatial densities.

    Returns:
        Tensor of shape [B, N, 2], ordered as (x_norm, y_norm) in [-1, 1].
    """
    with torch.no_grad():
        b, _, h, w = density.shape
        prob = normalize_sum(density).reshape(b, h * w)
        idx = torch.multinomial(prob, num_points, replacement=True)

        yy = torch.div(idx, w, rounding_mode="floor").float()
        xx = (idx % w).float()
        xx = xx + torch.rand_like(xx)
        yy = yy + torch.rand_like(yy)

        x_norm = 2.0 * xx / max(w - 1, 1) - 1.0
        y_norm = 2.0 * yy / max(h - 1, 1) - 1.0
        return torch.stack([x_norm, y_norm], dim=-1).clamp(-1, 1)


def _sample_field(field: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Bilinearly sample a field [B,C,H,W] at points [B,N,2]."""
    grid = points[:, :, None, :]
    value = F.grid_sample(field, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return value[:, :, :, 0].transpose(1, 2)


def phase_gradient_flow_loss(
    velocity_phases: list[torch.Tensor],
    rho0: torch.Tensor,
    target: torch.Tensor,
    k: float,
    dx: float,
    dz: float,
    num_samples: int = 256,
    target_floor: float = 1e-4,
    blur_sigma: float = 0.6,
) -> torch.Tensor:
    """Flow-matching loss on phase-induced optical velocity fields.

    A point sampled from the initial optical density and a point sampled from
    the target image density define a simple displacement path. We then match
    the normalized phase-gradient velocity along this path:

        v_z(r) = grad_perp phi_z(r) / k.

    This term directly supervises the optical velocity field.
    """
    if not velocity_phases:
        return torch.tensor(0.0, device=rho0.device)

    target_density = prepare_density_target(target, target_floor, blur_sigma)
    x0 = _sample_points_from_density(rho0.detach(), num_samples).to(rho0.device)
    x1 = _sample_points_from_density(target_density.detach(), num_samples).to(rho0.device)

    total_z = max(len(velocity_phases) * dz, 1e-12)
    target_velocity_norm = (x1 - x0) / total_z

    _, _, h, w = rho0.shape
    scale_x = (w - 1) * dx / 2.0
    scale_y = (h - 1) * dx / 2.0

    loss = 0.0
    for layer, phase in enumerate(velocity_phases):
        alpha = (layer + 0.5) / len(velocity_phases)
        xz = ((1.0 - alpha) * x0 + alpha * x1).clamp(-1, 1)

        gx, gy = phase_gradient(phase, dx)
        grad = torch.cat([gx, gy], dim=1)
        velocity_phys = _sample_field(grad, xz) / k
        velocity_norm = torch.stack(
            [velocity_phys[..., 0] / scale_x, velocity_phys[..., 1] / scale_y],
            dim=-1,
        )
        loss = loss + F.mse_loss(velocity_norm, target_velocity_norm)

    return loss / len(velocity_phases)


def _central_diff_x(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (torch.roll(x, -1, dims=-1) - torch.roll(x, 1, dims=-1))


def _central_diff_y(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (torch.roll(x, -1, dims=-2) - torch.roll(x, 1, dims=-2))


def tie_residual_loss(
    intensities: list[torch.Tensor],
    velocity_phases: list[torch.Tensor],
    k: float,
    dx: float,
    dz: float,
) -> torch.Tensor:
    """Discrete TIE residual for physics consistency.

    The residual is evaluated in dimensionless pixel units for numerical
    stability, but still follows the TIE structure:

        d rho / dz + div(rho grad(phi)/k) = 0.
    """
    if len(intensities) < 2 or not velocity_phases:
        return torch.tensor(0.0, device=intensities[0].device)

    loss = 0.0
    for rho_l, rho_next, phase in zip(intensities[:-1], intensities[1:], velocity_phases):
        gx, gy = phase_gradient(phase, dx)
        vx = gx / k / dx
        vy = gy / k / dx
        flux_x = rho_l * vx
        flux_y = rho_l * vy
        div = _central_diff_x(flux_x) + _central_diff_y(flux_y)
        drho = (rho_next - rho_l) / max(dz, 1e-12)
        residual = drho + div
        loss = loss + residual.pow(2).mean()
    return loss / len(velocity_phases)
