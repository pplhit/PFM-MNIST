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
    density_loss,
    diversity_loss,
    feature_matching_loss,
    hinge_d_loss,
    hinge_g_loss,
    phase_smoothness_loss,
)
from pfm_mnist.models import ConditionalDiscriminator, PFMGenerator
from pfm_mnist.utils import ensure_dir, make_noisy_seed, normalize_for_display, read_yaml, seed_everything


def build_generator(config: dict[str, Any]) -> PFMGenerator:
    return PFMGenerator(**config["model"])


def save_checkpoint(
    run_dir: Path,
    generator: PFMGenerator,
    discriminator: ConditionalDiscriminator,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    config: dict[str, Any],
    epoch: int,
    step: int,
) -> None:
    ckpt_dir = ensure_dir(run_dir / "checkpoints")
    state = {
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "opt_g": opt_g.state_dict(),
        "opt_d": opt_d.state_dict(),
        "config": config,
        "epoch": epoch,
        "step": step,
    }
    torch.save(state, ckpt_dir / "latest.pt")
    torch.save(state, ckpt_dir / f"epoch_{epoch:04d}.pt")


def train(config: dict[str, Any], args: argparse.Namespace) -> None:
    seed_everything(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
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
    discriminator = ConditionalDiscriminator(size=config["model"]["size"]).to(device)

    opt_g = torch.optim.AdamW(
        generator.parameters(),
        lr=float(config["train"]["lr_g"]),
        betas=(0.0, 0.99),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    opt_d = torch.optim.AdamW(
        discriminator.parameters(),
        lr=float(config["train"]["lr_d"]),
        betas=(0.0, 0.99),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(config["train"]["amp"]) and device.type == "cuda")

    fixed_labels = torch.arange(10, device=device).repeat_interleave(4)
    fixed_seed = torch.rand(fixed_labels.shape[0], 1, config["model"]["size"], config["model"]["size"], device=device)
    fixed_latent = generator.sample_latent(fixed_labels.shape[0], device=device)

    step = 0
    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        generator.train()
        discriminator.train()
        pbar = tqdm(loader, desc=f"epoch {epoch}/{config['train']['epochs']}")

        for real, labels in pbar:
            real = real.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            real_disp = normalize_for_display(real)

            # Discriminator update.
            for _ in range(int(config["train"]["d_steps"])):
                seed = make_noisy_seed(
                    real,
                    t_min=float(config["train"]["t_min"]),
                    pure_noise_prob=float(config["train"]["pure_noise_prob"]),
                )
                latent = generator.sample_latent(real.shape[0], device)
                with torch.no_grad():
                    fake = generator(seed, labels, latent=latent, return_fields=False)
                    fake_disp = normalize_for_display(fake)
                real_logits = discriminator(real_disp, labels)
                fake_logits = discriminator(fake_disp, labels)
                d_loss = hinge_d_loss(real_logits, fake_logits)
                opt_d.zero_grad(set_to_none=True)
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), float(config["train"]["grad_clip"]))
                opt_d.step()

            # Generator update.
            seed_a = make_noisy_seed(
                real,
                t_min=float(config["train"]["t_min"]),
                pure_noise_prob=float(config["train"]["pure_noise_prob"]),
            )
            seed_b = torch.rand_like(seed_a)
            latent_a = generator.sample_latent(real.shape[0], device)
            latent_b = generator.sample_latent(real.shape[0], device)

            with torch.cuda.amp.autocast(enabled=bool(config["train"]["amp"]) and device.type == "cuda"):
                fake_a, info_a = generator(seed_a, labels, latent=latent_a, return_fields=True)
                fake_b = generator(seed_b, labels, latent=latent_b, return_fields=False)
                fake_a_disp = normalize_for_display(fake_a)
                fake_logits, fake_features = discriminator(fake_a_disp, labels, return_features=True)
                _, real_features = discriminator(real_disp, labels, return_features=True)
                loss_adv = hinge_g_loss(fake_logits)
                loss_feature = feature_matching_loss(real_features, fake_features)
                loss_density = density_loss(
                    fake_a,
                    real,
                    target_floor=float(config["loss"]["target_floor"]),
                    blur_sigma=float(config["loss"]["target_blur_sigma"]),
                )
                loss_div = diversity_loss(
                    fake_a,
                    fake_b,
                    latent_a,
                    latent_b,
                    margin=float(config["loss"]["diversity_margin"]),
                )
                loss_anti = anti_collapse_loss(
                    fake_a,
                    real,
                    lambda_ipr=float(config["loss"]["lambda_ipr"]),
                    lambda_peak=float(config["loss"]["lambda_peak"]),
                    target_floor=float(config["loss"]["target_floor"]),
                    blur_sigma=float(config["loss"]["target_blur_sigma"]),
                )
                loss_tv = phase_smoothness_loss(info_a["phase_maps"])
                g_loss = (
                    float(config["loss"]["lambda_adv"]) * loss_adv
                    + float(config["loss"]["lambda_feature"]) * loss_feature
                    + float(config["loss"]["lambda_density"]) * loss_density
                    + float(config["loss"]["lambda_diversity"]) * loss_div
                    + loss_anti
                    + float(config["loss"]["lambda_phase_tv"]) * loss_tv
                )

            opt_g.zero_grad(set_to_none=True)
            scaler.scale(g_loss).backward()
            scaler.unscale_(opt_g)
            torch.nn.utils.clip_grad_norm_(generator.parameters(), float(config["train"]["grad_clip"]))
            scaler.step(opt_g)
            scaler.update()
            step += 1

            pbar.set_postfix(d=f"{d_loss.item():.3f}", g=f"{g_loss.item():.3f}", div=f"{loss_div.item():.3f}")

            if step % int(config["train"]["vis_every"]) == 0:
                generator.eval()
                with torch.no_grad():
                    fake, info = generator(fixed_seed, fixed_labels, latent=fixed_latent, return_fields=True)
                    grid = make_grid(normalize_for_display(fake), nrow=10, padding=2)
                    save_image(grid, image_dir / f"sample_step_{step:07d}.png")
                    phase = normalize_for_display(info["phase_maps"][:1])
                    phase_grid = make_grid(phase.transpose(0, 1), nrow=phase.shape[1], padding=2)
                    save_image(phase_grid, image_dir / f"phase_stack_step_{step:07d}.png")
                generator.train()

        if epoch % int(config["train"]["save_every"]) == 0:
            save_checkpoint(run_dir, generator, discriminator, opt_g, opt_d, config, epoch, step)

    save_checkpoint(run_dir, generator, discriminator, opt_g, opt_d, config, epoch, step)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train diverse PFM on MNIST")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lambda-diversity", type=float, default=None)
    parser.add_argument("--lambda-adv", type=float, default=None)
    parser.add_argument("--pure-noise-prob", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_yaml(args.config)
    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs
    if args.lambda_diversity is not None:
        config["loss"]["lambda_diversity"] = args.lambda_diversity
    if args.lambda_adv is not None:
        config["loss"]["lambda_adv"] = args.lambda_adv
    if args.pure_noise_prob is not None:
        config["train"]["pure_noise_prob"] = args.pure_noise_prob
    train(config, args)


if __name__ == "__main__":
    main()
