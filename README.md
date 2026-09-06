# PFM-MNIST

A PyTorch implementation of Photonic Flow Matching (PFM) for diverse MNIST generation.

This repository implements a hybrid optoelectronic generator:

```text
text/label + noise + latent code -> input-dependent phase stack -> angular-spectrum propagation -> optical intensity image
```

The current version improves diversity over a purely reconstruction-based implementation by adding conditional adversarial training, feature matching, latent diversity regularization, density losses, and anti-collapse optical penalties.
