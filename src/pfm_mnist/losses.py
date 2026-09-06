from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F

from pfm_mnist.utils import gaussian_blur, normalize_for_display, normalize_sum, total_variation


def hinge_d_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    return F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()


def hinge_g_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    return -fake_logits.mean()


def feature_matching_loss(real_features: Iterable[torch.Tensor], fake_features: Iterable[torch.Tensor]) -> torch.Tensor:
    loss = 0.0
    for real, fake in zip(real_features, fake_features):
        loss = loss + F.l1_loss(fake.mean(dim=0), real.detach().mean(dim=0))
    return loss


def prepare_density_target(target: torch.Tensor, target_floor: float, blur_sigma: float) -> torch.Tensor:
    return normalize_sum(gaussian_blur(target, blur_sigma) + target_floor)


def density_loss(pred: torch.Tensor, target: torch.Tensor, target_floor: float, blur_sigma: float) -> torch.Tensor:
    eps = 1e-8
    pred = normalize_sum(pred + eps)
    target = prepare_density_target(target, target_floor, blur_sigma)
    hellinger = ((torch.sqrt(pred + eps) - torch.sqrt(target + eps)) ** 2).sum(dim=(-2, -1)).mean()
    kl_t2p = (target * (torch.log(target + eps) - torch.log(pred + eps))).sum(dim=(-2, -1)).mean()
    kl_p2t = (pred * (torch.log(pred + eps) - torch.log(target + eps))).sum(dim=(-2, -1)).mean()
    return hellinger + 0.05 * kl_t2p + 0.01 * kl_p2t


def anti_collapse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lambda_ipr: float,
    lambda_peak: float,
    target_floor: float,
    blur_sigma: float,
) -> torch.Tensor:
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
    margin: float,
) -> torch.Tensor:
    """Mode-seeking diversity loss for same-label different-latent samples."""
    a = gaussian_blur(normalize_for_display(out_a), sigma=1.2)
    b = gaussian_blur(normalize_for_display(out_b), sigma=1.2)
    image_dist = (a - b).abs().flatten(1).mean(dim=1)
    latent_dist = (latent_a - latent_b).abs().flatten(1).mean(dim=1).detach()
    return F.relu(margin - image_dist / (latent_dist + 1e-6)).mean()


def phase_smoothness_loss(phase_maps: torch.Tensor) -> torch.Tensor:
    return total_variation(phase_maps)
