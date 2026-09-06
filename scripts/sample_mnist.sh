#!/usr/bin/env bash
set -euo pipefail
python -m pfm_mnist.sample \
  --ckpt runs/pfm_mnist_nogan/checkpoints/latest.pt \
  --label all \
  --n-samples 40 \
  --out-dir runs/pfm_mnist_nogan/samples \
  "$@"
