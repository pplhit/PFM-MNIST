from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from pfm_mnist.data import make_mnist_loader
from pfm_mnist.losses import (
    anti_collapse_loss,
    diversity_loss,
    endpoint_density_loss,
    multiscale_density_loss,
    phase_gradient_flow_loss,
    phase_smoothness_loss,
    ssim_density_loss,
    tie_residual_loss,
)
from pfm_mnist.models import PFMGenerator
from pfm_mnist.utils import (
    ensure_dir,
    make_noisy_seed,
    normalize_for_display,
    read_yaml,
    seed_everything,
)


def build_generator(config: dict[str, Any]) -> PFMGenerator:
    return PFMGenerator(**config["model"])


def ramp_factor(epoch: int, start_epoch: int, ramp_epochs: int) -> float:
    """Linear warm-up factor in [0, 1]."""
    if epoch < start_epoch:
        return 0.0
    if ramp_epochs <= 0:
        return 1.0
    return min(1.0, max(0.0, (epoch - start_epoch + 1) / ramp_epochs))


def save_checkpoint(
    run_dir: Path,
    generator: PFMGenerator,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    epoch: int,
    step: int,
) -> None:
    ckpt_dir = ensure_dir(run_dir / "checkpoints")
    state = {
        "generator": generator.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
        "epoch": epoch,
        "step": step,
    }
    torch.save(state, ckpt_dir / "latest.pt")
    torch.save(state, ckpt_dir / f"epoch_{epoch:04d}.pt")


def compute_pfm_loss(
    generator: PFMGenerator,
    pred_a: torch.Tensor,
    pred_b: torch.Tensor,
    real: torch.Tensor,
    info_a: dict[str, Any],
    latent_a: torch.Tensor,
    latent_b: torch.Tensor,
    config: dict[str, Any],
    epoch: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    loss_cfg = config["loss"]
    train_cfg = config["train"]

    physics_ramp = ramp_factor(
        epoch,
        start_epoch=int(train_cfg.get("physics_start_epoch", 1)),
        ramp_epochs=int(train_cfg.get("physics_ramp_epochs", 1)),
    )
    diversity_ramp = ramp_factor(
        epoch,
        start_epoch=1,
        ramp_epochs=int(train_cfg.get("diversity_ramp_epochs", 1)),
    )

    lambda_flow_eff = float(loss_cfg.get("lambda_flow", 0.0)) * physics_ramp
    lambda_tie_eff = float(loss_cfg.get("lambda_tie", 0.0)) * physics_ramp
    lambda_div_eff = float(loss_cfg.get("lambda_diversity", 0.0)) * diversity_ramp

    l_density = endpoint_density_loss(
        pred_a,
        real,
        target_floor=float(loss_cfg["target_floor"]),
        blur_sigma=float(loss_cfg["target_blur_sigma"]),
    )
    l_ms = multiscale_density_loss(
        pred_a,
        real,
        target_floor=float(loss_cfg["target_floor"]),
        blur_sigma=float(loss_cfg["target_blur_sigma"]),
    )
    l_ssim = ssim_density_loss(
        pred_a,
        real,
        target_floor=float(loss_cfg["target_floor"]),
        blur_sigma=float(loss_cfg["target_blur_sigma"]),
    )

    if lambda_flow_eff > 0.0:
        l_flow = phase_gradient_flow_loss(
            velocity_phases=info_a["velocity_phases"],
            rho0=info_a["rho0"],
            target=real,
            k=generator.k,
            dx=generator.dx,
            dz=generator.dz,
            num_samples=int(loss_cfg["flow_samples"]),
            target_floor=float(loss_cfg["target_floor"]),
            blur_sigma=float(loss_cfg["target_blur_sigma"]),
        )
    else:
        l_flow = pred_a.new_tensor(0.0)

    if lambda_tie_eff > 0.0:
        l_tie = tie_residual_loss(
            intensities=info_a["intensities"],
            velocity_phases=info_a["velocity_phases"],
            k=generator.k,
            dx=generator.dx,
            dz=generator.dz,
        )
    else:
        l_tie = pred_a.new_tensor(0.0)

    if lambda_div_eff > 0.0:
        l_div = diversity_loss(
            pred_a,
            pred_b,
            latent_a,
            latent_b,
            margin=float(loss_cfg["diversity_margin"]),
        )
    else:
        l_div = pred_a.new_tensor(0.0)

    l_anti = anti_collapse_loss(
        pred_a,
        real,
        lambda_ipr=float(loss_cfg["lambda_ipr"]),
        lambda_peak=float(loss_cfg["lambda_peak"]),
        target_floor=float(loss_cfg["target_floor"]),
        blur_sigma=float(loss_cfg["target_blur_sigma"]),
    )
    l_tv = phase_smoothness_loss(info_a["phase_maps"])

    total = (
        float(loss_cfg["lambda_density"]) * l_density
        + float(loss_cfg["lambda_multiscale"]) * l_ms
        + float(loss_cfg.get("lambda_ssim", 0.0)) * l_ssim
        + lambda_flow_eff * l_flow
        + lambda_tie_eff * l_tie
        + lambda_div_eff * l_div
        + l_anti
        + float(loss_cfg["lambda_phase_tv"]) * l_tv
    )

    stats = {
        "loss": float(total.detach().cpu()),
        "density": float(l_density.detach().cpu()),
        "multiscale": float(l_ms.detach().cpu()),
        "ssim": float(l_ssim.detach().cpu()),
        "flow": float(l_flow.detach().cpu()),
        "tie": float(l_tie.detach().cpu()),
        "diversity": float(l_div.detach().cpu()),
        "anti": float(l_anti.detach().cpu()),
        "phase_tv": float(l_tv.detach().cpu()),
        "lambda_flow_eff": lambda_flow_eff,
        "lambda_tie_eff": lambda_tie_eff,
        "lambda_div_eff": lambda_div_eff,
    }
    return total, stats


def train(config: dict[str, Any], overrides: argparse.Namespace) -> None:
    seed_everything(int(config["seed"]))

    device = torch.device("cuda" if torch.cuda.is_available() and not overrides.cpu else "cpu")
    run_dir = ensure_dir(config["run_dir"])
    image_dir = ensure_dir(run_dir / "images")

    loader = make_mnist_loader(
        data_root=config["data_root"],
        size=config["model"]["size"],
        batch_size=config["train"]["batch_size"],
        train=True,
        num_workers=config["train"]["num_workers"],
        shuffle=True,
    )

    generator = build_generator(config).to(device)
    optimizer = torch.optim.AdamW(
        generator.parameters(),
        lr=float(config["train"]["lr"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(config["train"]["amp"]) and device.type == "cuda")

    fixed_labels = torch.arange(10, device=device).repeat_interleave(4)
    fixed_seed = torch.rand(
        fixed_labels.shape[0],
        1,
        config["model"]["size"],
        config["model"]["size"],
        device=device,
    )
    fixed_latent = generator.sample_latent(fixed_labels.shape[0], device=device)

    step = 0
    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        generator.train()
        pbar = tqdm(loader, desc=f"epoch {epoch}/{config['train']['epochs']}")

        for real, labels in pbar:
            real = real.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            seed_a = make_noisy_seed(
                real,
                t_min=float(config["train"]["t_min"]),
                pure_noise_prob=float(config["train"]["pure_noise_prob"]),
            )
            seed_b = torch.rand_like(seed_a)
            latent_a = generator.sample_latent(real.shape[0], device)
            latent_b = generator.sample_latent(real.shape[0], device)

            with torch.cuda.amp.autocast(enabled=bool(config["train"]["amp"]) and device.type == "cuda"):
                pred_a, info_a = generator(seed_a, labels, latent=latent_a, return_fields=True)
                pred_b = generator(seed_b, labels, latent=latent_b, return_fields=False)
                loss, stats = compute_pfm_loss(
                    generator=generator,
                    pred_a=pred_a,
                    pred_b=pred_b,
                    real=real,
                    info_a=info_a,
                    latent_a=latent_a,
                    latent_b=latent_b,
                    config=config,
                    epoch=epoch,
                )

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(generator.parameters(), float(config["train"]["grad_clip"]))
            scaler.step(optimizer)
            scaler.update()

            step += 1
            pbar.set_postfix(
                loss=f"{stats['loss']:.3f}",
                den=f"{stats['density']:.3f}",
                ssim=f"{stats['ssim']:.3f}",
                flow_w=f"{stats['lambda_flow_eff']:.1e}",
                div_w=f"{stats['lambda_div_eff']:.2f}",
            )

            if step % int(config["train"]["vis_every"]) == 0:
                generator.eval()
                with torch.no_grad():
                    pred, info = generator(
                        fixed_seed,
                        fixed_labels,
                        latent=fixed_latent,
                        return_fields=True,
                    )
                    grid = make_grid(normalize_for_display(pred), nrow=10, padding=2)
                    save_image(grid, image_dir / f"sample_step_{step:07d}.png")

                    # Visualize phase stacks for several inputs. Rows correspond to
                    # different noise/latent inputs; columns correspond to phi0, psi1, ...
                    phase = normalize_for_display(info["phase_maps"][:8])
                    b, p, h, w = phase.shape
                    phase_grid = make_grid(phase.reshape(b * p, 1, h, w), nrow=p, padding=2)
                    save_image(phase_grid, image_dir / f"phase_stacks_step_{step:07d}.png")
                generator.train()

        if epoch % int(config["train"]["save_every"]) == 0:
            save_checkpoint(run_dir, generator, optimizer, config, epoch, step)

    save_checkpoint(run_dir, generator, optimizer, config, epoch, step)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train PFM on MNIST")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--lambda-diversity", type=float, default=None)
    parser.add_argument("--lambda-flow", type=float, default=None)
    parser.add_argument("--lambda-tie", type=float, default=None)
    parser.add_argument("--pure-noise-prob", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_yaml(args.config)

    if args.lambda_diversity is not None:
        config["loss"]["lambda_diversity"] = args.lambda_diversity
    if args.lambda_flow is not None:
        config["loss"]["lambda_flow"] = args.lambda_flow
    if args.lambda_tie is not None:
        config["loss"]["lambda_tie"] = args.lambda_tie
    if args.pure_noise_prob is not None:
        config["train"]["pure_noise_prob"] = args.pure_noise_prob
    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs

    train(config, args)


if __name__ == "__main__":
    main()
