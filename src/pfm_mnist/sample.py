from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torchvision.utils import make_grid, save_image

from pfm_mnist.models import PFMGenerator
from pfm_mnist.photon import integrate_photon_flow, photon_histogram
from pfm_mnist.utils import ensure_dir, normalize_for_display, parse_label


def load_generator(path: str | Path, device: torch.device) -> tuple[PFMGenerator, dict]:
    ckpt = torch.load(path, map_location=device)
    config = ckpt["config"]
    model = PFMGenerator(**config["model"]).to(device)
    model.load_state_dict(ckpt["generator"], strict=True)
    model.eval()
    return model, config


@torch.no_grad()
def sample(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    out_dir = ensure_dir(args.out_dir)
    model, _ = load_generator(args.ckpt, device)
    size = model.size

    if args.label == "all":
        labels = torch.arange(10, device=device).repeat_interleave(max(1, args.n_samples // 10 + 1))
        labels = labels[: args.n_samples]
    else:
        labels = torch.full((args.n_samples,), parse_label(args.label), device=device, dtype=torch.long)

    seed = torch.rand(args.n_samples, 1, size, size, device=device)
    latent = model.sample_latent(args.n_samples, device)
    out, info = model(seed, labels, latent=latent, return_fields=True)

    grid = make_grid(normalize_for_display(out), nrow=max(1, int(math.sqrt(args.n_samples))), padding=2)
    save_image(grid, out_dir / "generated_grid.png")

    one_out = out[:1]
    one_info = {k: (v[:1] if torch.is_tensor(v) else v) for k, v in info.items()}
    save_image(normalize_for_display(one_info["rho0"]), out_dir / "rho0.png")
    save_image(normalize_for_display(one_out), out_dir / "output_intensity.png")

    phase_maps = normalize_for_display(one_info["phase_maps"])
    for idx in range(phase_maps.shape[1]):
        name = "phi0" if idx == 0 else f"psi_{idx}"
        save_image(phase_maps[:, idx : idx + 1], out_dir / f"{name}.png")

    for idx, intensity in enumerate(info["intensities"]):
        save_image(normalize_for_display(intensity[:1]), out_dir / f"intensity_plane_{idx}.png")

    if args.photon_decode:
        xy = integrate_photon_flow(model, one_info, n_photons=args.n_photons)
        hist = photon_histogram(xy, size=size, dx=model.dx, device=device)
        save_image(normalize_for_display(hist), out_dir / "lagrangian_photon_histogram.png")

    print(f"Saved samples to {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Sample from a trained PFM-MNIST generator")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--label", type=str, default="all")
    parser.add_argument("--n-samples", type=int, default=40)
    parser.add_argument("--out-dir", type=str, default="runs/pfm_mnist/samples")
    parser.add_argument("--photon-decode", action="store_true")
    parser.add_argument("--n-photons", type=int, default=100000)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    sample(parse_args())


if __name__ == "__main__":
    main()
