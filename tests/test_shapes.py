import torch

from pfm_mnist.losses import phase_gradient_flow_loss, tie_residual_loss
from pfm_mnist.models import PFMGenerator


def test_generator_shapes():
    model = PFMGenerator(size=32, num_layers=2, base_channels=16, latent_dim=8, text_dim=8)
    seed = torch.rand(4, 1, 32, 32)
    labels = torch.tensor([0, 1, 2, 3])
    latent = torch.randn(4, 8)

    out, info = model(seed, labels, latent=latent, return_fields=True)
    assert out.shape == (4, 1, 32, 32)
    assert info["phase_maps"].shape == (4, 3, 32, 32)
    assert len(info["intensities"]) == 3
    assert len(info["velocity_phases"]) == 2


def test_pfm_losses_run():
    model = PFMGenerator(size=32, num_layers=2, base_channels=16, latent_dim=8, text_dim=8)
    seed = torch.rand(2, 1, 32, 32)
    target = torch.rand(2, 1, 32, 32)
    labels = torch.tensor([0, 1])
    latent = torch.randn(2, 8)

    _, info = model(seed, labels, latent=latent, return_fields=True)
    flow = phase_gradient_flow_loss(
        info["velocity_phases"],
        info["rho0"],
        target,
        k=model.k,
        dx=model.dx,
        dz=model.dz,
        num_samples=16,
    )
    tie = tie_residual_loss(
        info["intensities"],
        info["velocity_phases"],
        k=model.k,
        dx=model.dx,
        dz=model.dz,
    )
    assert torch.isfinite(flow)
    assert torch.isfinite(tie)
