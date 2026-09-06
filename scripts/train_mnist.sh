#!/usr/bin/env bash
set -euo pipefail
python -m pfm_mnist.train --config configs/default.yaml "$@"
