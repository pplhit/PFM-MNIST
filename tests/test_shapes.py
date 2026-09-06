import torch

from pfm_mnist.models import ConditionalDiscriminator, PFMGenerator


def test_generator_shapes():
    model = PFMGenerator(size=32, num_layers=2, base_channels=16, latent_dim=8, text_dim=8)
    seed = torch.rand(4, 1, 32, 32)
    labels = torch.tensor([0, 1, 2, 3])
    latent = torch.randn(4, 8)
    out, info = model(seed, labels, latent=latent, return_fields=True)
    assert out.shape == (4, 1, 32, 32)
    assert info["phase_maps"].shape == (4, 3, 32, 32)
    assert len(info["intensities"]) == 3


def test_discriminator_shapes():
    disc = ConditionalDiscriminator(size=32, base_channels=16)
    x = torch.rand(4, 1, 32, 32)
    labels = torch.tensor([0, 1, 2, 3])
    logits = disc(x, labels)
    assert logits.shape == (4, 1)
