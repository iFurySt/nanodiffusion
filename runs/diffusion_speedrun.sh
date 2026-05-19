#!/usr/bin/env bash
set -euo pipefail

# Minimal masked diffusion base training run.
# Prepare the dataset/tokenizer first with the existing nanochat pipeline:
#   python -m nanochat.dataset -n 10
#   python -m scripts.tok_train --max-chars=2000000000

torchrun --nproc_per_node=8 -m scripts.diffusion_base_train -- \
  --run=diffusion-speedrun \
  --depth=20 \
  --max-seq-len=2048 \
  --device-batch-size=16 \
  --total-batch-size=524288 \
  --target-param-data-ratio=12 \
  --eval-every=250 \
  --eval-batches=20 \
  --save-every=2000 \
  --compile
