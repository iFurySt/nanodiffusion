#!/bin/bash

# A100 variant of runs/speedrun.sh.
# The reference speedrun is tuned for 8xH100 and enables FP8. On 8xA100-80G,
# keep bf16 compute and omit --fp8. Set NANOCHAT_BASE_DIR to point at a large
# disk or an existing cache before launching.

set -euo pipefail

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$NANOCHAT_BASE_DIR/uv-cache}"
export HF_HOME="${HF_HOME:-$NANOCHAT_BASE_DIR/huggingface}"
mkdir -p "$NANOCHAT_BASE_DIR"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-16}"
WINDOW_PATTERN="${WINDOW_PATTERN:-L}"

command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
if [ ! -d ".venv" ]; then
    if [ -n "${NANOCHAT_UV_PYTHON:-}" ]; then
        uv venv --python "$NANOCHAT_UV_PYTHON"
    else
        uv venv
    fi
fi
uv sync --extra gpu
source .venv/bin/activate

if [ -z "${WANDB_RUN:-}" ]; then
    WANDB_RUN=dummy
fi

python -m nanochat.report reset

python -m nanochat.dataset -n 8
python -m nanochat.dataset -n 170 &
DATASET_DOWNLOAD_PID=$!
python -m scripts.tok_train
python -m scripts.tok_eval

echo "Waiting for dataset download to complete..."
wait "$DATASET_DOWNLOAD_PID"

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_train -- \
    --depth=24 \
    --target-param-data-ratio=8 \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --window-pattern="$WINDOW_PATTERN" \
    --run="$WANDB_RUN"
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_eval -- \
    --device-batch-size="$DEVICE_BATCH_SIZE"

curl -L -o "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" \
    https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.chat_sft -- \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --run="$WANDB_RUN"
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.chat_eval -- -i sft

python -m nanochat.report generate
