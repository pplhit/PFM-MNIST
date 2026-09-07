from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from pfm_mnist.optics import phase_gradient
from pfm_mnist.utils import gaussian_blur, normalize_for_display, normalize_sum, total_variation


def prepare_density_target(
    target: torch.Tensor,
    target_floor: float = 1e-4,
    blur_sigma: float = 0.6,
) -> torch.Tensor:
    """Convert a target image into a normalized photon-density target."""
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

    Hellinger distance gives a stable probability-density objective, while weak
    forward/reverse KL terms make the predicted density cover the digit strokes
    without allowing large background hot spots.
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
    """Multi-scale image-domain density loss for global digit geometry."""
    pred = normalize_for_display(normalize_sum(pred))
    target = normalize_for_display(prepare_density_target(target, target_floor, blur_sigma))

    loss = 0.0
    for sigma, weight in [(0.0, 1.0), (1.0, 0.7), (2.0, 0.5), (4.0, 0.3)]:
        loss = loss + weight * F.l1_loss(gaussian_blur(pred, sigma), gaussian_blur(target, sigma))
    return loss


def ssim_density_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    target_floor: float = 1e-4,
    blur_sigma: float = 0.6,
    window: int = 7,
) -> torch.Tensor:
    """SSIM-style structural loss restored from the stable v2 script.

    MNIST is a structured sparse image. This term strongly improves digit
    readability without changing the PFM optical forward model.
    """
    eps = 1e-8
    pred = normalize_for_display(normalize_sum(pred))
    target = normalize_for_display(prepare_density_target(target, target_floor, blur_sigma))

    pad = window // 2
    mu_x = F.avg_pool2d(pred, window, stride=1, padding=pad)
    mu_y = F.avg_pool2d(target, window, stride=1, padding=pad)

    sigma_x = F.avg_pool2d(pred * pred, window, stride=1, padding=pad) - mu_x * mu_x
    sigma_y = F.avg_pool2d(target * target, window, stride=1, padding=pad) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(pred * target, window, stride=1, padding=pad) - mu_x * mu_y

    c1 = 0.01**2
    c2 = 0.03**2
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2) + eps
    )
    return 1.0 - ssim.mean()


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
    different photon-density patterns. The loss is moderate by default and is
    ramped in training so it does not destroy early digit formation.
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
        [B, N, 2], ordered as (x_norm, y_norm) in [-1, 1].
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


def _projection_coupling(x0: torch.Tensor, x1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Cheap sliced-transport coupling for flow supervision.

    Independent random pairing produces a very noisy target velocity field. We
    therefore sort both point clouds along a random projection and pair points
    with the same rank. This is not a full OT solver, but it is a much smoother
    approximation of a monotone transport plan and works well for MNIST.
    """
    b, _, _ = x0.shape
    direction = torch.randn(b, 2, device=x0.device, dtype=x0.dtype)
    direction = direction / (direction.norm(dim=1, keepdim=True) + 1e-8)

    p0 = (x0 * direction[:, None, :]).sum(dim=-1)
    p1 = (x1 * direction[:, None, :]).sum(dim=-1)
    order0 = p0.argsort(dim=1)
    order1 = p1.argsort(dim=1)

    gather0 = order0[..., None].expand_as(x0)
    gather1 = order1[..., None].expand_as(x1)
    return torch.gather(x0, 1, gather0), torch.gather(x1, 1, gather1)


def _sample_coupled_points(
    rho0: torch.Tensor,
    target_density: torch.Tensor,
    num_points: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    x0 = _sample_points_from_density(rho0.detach(), num_points).to(rho0.device)
    x1 = _sample_points_from_density(target_density.detach(), num_points).to(rho0.device)
    return _projection_coupling(x0, x1)


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
    num_samples: int = 128,
    target_floor: float = 1e-4,
    blur_sigma: float = 0.6,
) -> torch.Tensor:
    """PFM velocity loss on the phase-induced optical velocity field.

    We construct a displacement path from an initial photon-density sample x0
    to a target-density sample x1:

        x_z = (1 - alpha) x0 + alpha x1,
        v_target = (x1 - x0) / (L dz).

    The sampled points are paired by a sliced-transport coupling rather than a
    purely random pairing, which makes the target velocity smoother. The model
    velocity is the normalized phase-gradient velocity:

        v_pfm(r) = grad_perp phi(r) / k.
    """
    if not velocity_phases:
        return torch.tensor(0.0, device=rho0.device)

    target_density = prepare_density_target(target, target_floor, blur_sigma)
    x0, x1 = _sample_coupled_points(rho0, target_density, num_samples)

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

    Instead of dividing by dz and dx separately, we use a stable finite-step
    continuity residual in pixel units:

        rho_{l+1} - rho_l + div_pixel(rho_l * Delta x_pixel) = 0,
        Delta x_pixel = dz * grad(phi) / (k dx).

    This keeps the TIE term useful as a weak physics regularizer without
    overwhelming the endpoint image objective.
    """
    if len(intensities) < 2 or not velocity_phases:
        return torch.tensor(0.0, device=intensities[0].device)

    loss = 0.0
    for rho_l, rho_next, phase in zip(intensities[:-1], intensities[1:], velocity_phases):
        gx, gy = phase_gradient(phase, dx)
        disp_x_pix = dz * gx / k / dx
        disp_y_pix = dz * gy / k / dx
        flux_x = rho_l * disp_x_pix
        flux_y = rho_l * disp_y_pix
        div = _central_diff_x(flux_x) + _central_diff_y(flux_y)
        residual = (rho_next - rho_l) + div
        loss = loss + residual.pow(2).mean()
    return loss / len(velocity_phases)
