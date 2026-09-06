# PFM-MNIST

A GitHub-ready PyTorch implementation of **Photonic Flow Matching (PFM)** for MNIST generation.

This repository deliberately keeps the training objective in the **flow-matching / photon-density-transport** family. It does **not** use GAN loss or a discriminator. The core process is:

```text
label/text condition + random optical seed + latent code
        -> input-dependent phase stack
        -> phase-gradient velocity fields
        -> angular-spectrum optical propagation
        -> output photon-density image
```

The generated image is the normalized output optical intensity:

```math
I_Z(\mathbf r)=|U_Z(\mathbf r)|^2,\qquad
\rho_Z(\mathbf r)=\frac{I_Z(\mathbf r)}{\int I_Z(\mathbf r)d\mathbf r}.
```

The corresponding PFM velocity interpretation is:

```math
\mathbf v_z(\mathbf r)=\frac{1}{k}\nabla_\perp\phi_z(\mathbf r),\qquad
X_Z=X_0+\int_0^Z \mathbf v_s(X_s)ds.
```

## Why this version removes GAN loss

PFM is intended to be a photonic flow-matching paradigm, not an optical GAN. A discriminator can improve visual realism as an engineering trick, but it changes the conceptual training objective. This code therefore uses only PFM-consistent losses:

- endpoint photon-density loss;
- multi-scale image-density loss;
- phase-gradient flow-matching loss;
- discrete TIE residual loss;
- latent-sensitive diversity loss;
- peak/IPR anti-collapse loss;
- phase smoothness regularization.

## Installation

```bash
git clone https://github.com/pplhit/PFM-MNIST.git
cd PFM-MNIST
pip install -e .
```

## Train

```bash
python -m pfm_mnist.train --config configs/default.yaml
```

For stronger diversity:

```bash
python -m pfm_mnist.train \
  --config configs/default.yaml \
  --lambda-diversity 0.25 \
  --pure-noise-prob 0.60
```

## Sample

```bash
python -m pfm_mnist.sample \
  --ckpt runs/pfm_mnist_nogan/checkpoints/latest.pt \
  --label all \
  --n-samples 40 \
  --out-dir runs/pfm_mnist_nogan/samples
```

Photon-flow decoding:

```bash
python -m pfm_mnist.sample \
  --ckpt runs/pfm_mnist_nogan/checkpoints/latest.pt \
  --label three \
  --n-samples 4 \
  --photon-decode \
  --n-photons 200000 \
  --out-dir runs/pfm_mnist_nogan/photon
```

## Project structure

```text
PFM-MNIST/
├── configs/default.yaml
├── scripts/train_mnist.sh
├── scripts/sample_mnist.sh
├── src/pfm_mnist/
│   ├── data.py
│   ├── losses.py
│   ├── models.py
│   ├── optics.py
│   ├── photon.py
│   ├── sample.py
│   ├── train.py
│   └── utils.py
└── tests/test_shapes.py
```
