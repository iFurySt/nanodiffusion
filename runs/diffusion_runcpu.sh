#!/usr/bin/env bash
set -euo pipefail

# Tiny end-to-end NanoDiffusion run for CPU/MPS laptops.
# This is for codepath validation and learning, not model quality.

export NANODIFFUSION_BASE_DIR="${NANODIFFUSION_BASE_DIR:-$HOME/.cache/nanodiffusion-cpu}"

echo "Using NANODIFFUSION_BASE_DIR=$NANODIFFUSION_BASE_DIR"

# Download one train shard plus the validation shard.
python -m nanochat.dataset -n 1 -w 2

# Train a small tokenizer quickly from the downloaded shard.
python -m scripts.tok_train \
  --max-chars=20000000 \
  --vocab-size=2048

# Train a tiny bidirectional denoiser.
python -m scripts.diffusion_base_train \
  --device-type=cpu \
  --depth=2 \
  --aspect-ratio=16 \
  --head-dim=16 \
  --max-seq-len=128 \
  --device-batch-size=2 \
  --total-batch-size=256 \
  --num-iterations=20 \
  --eval-every=10 \
  --eval-batches=2 \
  --model-tag=diffusion_cpu \
  --warmup-steps=2

# Sample from the tiny model.
python -m scripts.diffusion_base_eval \
  --device-type=cpu \
  --model-tag=diffusion_cpu \
  --eval=sample \
  --prompt="The capital of France is" \
  --max-tokens=16 \
  --steps=16
