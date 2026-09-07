# PFM-MNIST

A GitHub-ready PyTorch implementation of **Photonic Flow Matching (PFM)** for MNIST generation.

This repository keeps the training objective inside the **flow-matching / photon-density-transport** formulation. The core process is:

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

## Current training design

The current version balances two goals:

1. **Good MNIST generation quality.** The main losses follow the stable `pfm_mnist_v2.py` behavior: endpoint photon-density loss, multi-scale density loss, SSIM-style structural loss, peak/IPR anti-collapse loss, and phase smoothness regularization.
2. **PFM theoretical consistency.** Phase-gradient flow matching and TIE residual losses are still implemented, but they are introduced only after a warm-up period and with small weights. This avoids the earlier failure mode where strong physics losses damaged image quality before the optical endpoint mapping had formed.

The total objective is:

```math
\mathcal L=\lambda_d\mathcal L_{density}
+\lambda_m\mathcal L_{multi}
+\lambda_s\mathcal L_{SSIM}
+\lambda_f(t)\mathcal L_{flow}
+\lambda_{TIE}(t)\mathcal L_{TIE}
+\lambda_{div}(t)\mathcal L_{div}
+\mathcal L_{anti-collapse}
+\lambda_{TV}\mathcal L_{phase}.
```

The effective weights \(\lambda_f(t)\), \(\lambda_{TIE}(t)\), and \(\lambda_{div}(t)\) are ramped during training.

## Important implementation details

- The phase generator outputs one input-dependent phase stack per sample:

```math
\Phi_i=[\phi_{0,i},\psi_{1,i},\ldots,\psi_{L,i}].
```

- The optical forward model is differentiable angular-spectrum propagation:

```math
U_{l+1}=\mathcal F^{-1}\{\mathcal F[U_l\exp(i\psi_l)]H_{\Delta z}\}.
```

- The flow loss uses a cheap sliced-transport coupling instead of random point pairing. This gives a smoother target velocity:

```math
x_z=(1-\alpha)x_0+\alpha x_1,\qquad
v_{target}=\frac{x_1-x_0}{L\Delta z}.
```

- The TIE residual is written as a stable finite-step continuity residual:

```math
\rho_{l+1}-\rho_l+
\nabla_{pixel}\cdot\left(\rho_l\,\Delta x_{pixel}\right)=0,
\qquad
\Delta x_{pixel}=\frac{\Delta z}{k\Delta x}\nabla_\perp\phi_l.
```

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

For stronger diversity after the model can generate clear digits:

```bash
python -m pfm_mnist.train \
  --config configs/default.yaml \
  --lambda-diversity 0.12 \
  --pure-noise-prob 0.20
```

For a more theory-heavy run, increase the PFM physics weights carefully:

```bash
python -m pfm_mnist.train \
  --config configs/default.yaml \
  --lambda-flow 0.002 \
  --lambda-tie 0.0002
```

## Sample

```bash
python -m pfm_mnist.sample \
  --ckpt runs/pfm_mnist/checkpoints/latest.pt \
  --label all \
  --n-samples 40 \
  --out-dir runs/pfm_mnist/samples
```

Photon-flow decoding:

```bash
python -m pfm_mnist.sample \
  --ckpt runs/pfm_mnist/checkpoints/latest.pt \
  --label three \
  --n-samples 4 \
  --photon-decode \
  --n-photons 200000 \
  --out-dir runs/pfm_mnist/photon
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
