#!/usr/bin/env bash
set -euo pipefail

# Instruction-tune a NanoDiffusion base checkpoint with response-only masking.
# The script keeps prompt/user tokens fixed and trains only assistant answer
# tokens, matching the diffusion SFT path in scripts/diffusion_chat_sft.py.

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NANODIFFUSION_BASE_DIR="${NANODIFFUSION_BASE_DIR:-$HOME/.cache/nanodiffusion-a100}"
export HF_HOME="${HF_HOME:-$NANODIFFUSION_BASE_DIR/huggingface}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$NANODIFFUSION_BASE_DIR/uv-cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$NANODIFFUSION_BASE_DIR/pip-cache}"

PYTHON_BIN="${PYTHON_BIN:-}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
DEVICE_TYPE="${DEVICE_TYPE:-cuda}"
BASE_MODEL_TAG="${BASE_MODEL_TAG:-diffusion_a100_d20_s2048_5k}"
BASE_MODEL_STEP="${BASE_MODEL_STEP:-5000}"
OUTPUT_TAG="${OUTPUT_TAG:-${BASE_MODEL_TAG}_sft}"
RUN_NAME="${RUN_NAME:-$OUTPUT_TAG}"
SFT_DATA_PATH="${SFT_DATA_PATH:-$NANODIFFUSION_BASE_DIR/diffusion_sft_seed.jsonl}"
INCLUDE_SMOLTALK="${INCLUDE_SMOLTALK:-0}"
TRAIN_STEPS="${TRAIN_STEPS:-1000}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-4}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-65536}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
EVAL_EVERY="${EVAL_EVERY:-100}"
EVAL_BATCHES="${EVAL_BATCHES:-10}"
SAVE_EVERY="${SAVE_EVERY:-500}"
SAMPLE_MAX_TOKENS="${SAMPLE_MAX_TOKENS:-128}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
COMPILE="${COMPILE:-0}"

mkdir -p "$NANODIFFUSION_BASE_DIR/logs" "$NANODIFFUSION_BASE_DIR/report"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
REPORT_FILE="$NANODIFFUSION_BASE_DIR/report/${OUTPUT_TAG}-${TIMESTAMP}.md"
SFT_LOG="$NANODIFFUSION_BASE_DIR/logs/${OUTPUT_TAG}-${TIMESTAMP}.sft.log"

run_python() {
  if [ -n "$PYTHON_BIN" ]; then
    "$PYTHON_BIN" "$@"
  elif command -v uv >/dev/null 2>&1; then
    uv run python "$@"
  else
    python "$@"
  fi
}

append_report() {
  printf "%s\n" "$@" >> "$REPORT_FILE"
}

if [ ! -f "$SFT_DATA_PATH" ]; then
  mkdir -p "$(dirname "$SFT_DATA_PATH")"
  cat > "$SFT_DATA_PATH" <<'JSONL'
[{"role":"user","content":"Explain masked diffusion language models in one paragraph."},{"role":"assistant","content":"A masked diffusion language model learns to recover text whose tokens have been replaced by mask tokens. During training, it sees the full left and right context around each mask and predicts only the original masked tokens. During generation, it starts with a prompt plus masked answer slots, fills confident positions, and repeats until the answer window is complete."}]
[{"role":"user","content":"Write a tiny Python function that reverses a string."},{"role":"assistant","content":"def reverse_string(text):\n    return text[::-1]"}]
[{"role":"user","content":"Give three practical tips for training a small language model."},{"role":"assistant","content":"Use clean data, keep the run reproducible, and inspect fixed validation samples as well as loss. Start with a small model and short run, verify the full pipeline, then scale one variable at a time."}]
[{"role":"user","content":"What is NanoDiffusion?"},{"role":"assistant","content":"NanoDiffusion is a small educational codebase for training masked diffusion language models. It reuses nanochat-style data, tokenizer, checkpointing, and Transformer infrastructure while replacing next-token prediction with masked denoising."}]
JSONL
fi

torch_args=()
if [ "$COMPILE" = "1" ]; then
  torch_args+=(--compile)
fi
smoltalk_args=()
if [ "$INCLUDE_SMOLTALK" = "1" ]; then
  smoltalk_args+=(--include-smoltalk)
fi

commit="$(git rev-parse HEAD 2>/dev/null || cat .sync/source_commit 2>/dev/null || echo unknown)"
append_report "# NanoDiffusion SFT A100 Run"
append_report ""
append_report "- started: $(date)"
append_report "- commit: \`$commit\`"
append_report "- base_dir: \`$NANODIFFUSION_BASE_DIR\`"
append_report "- base_model: \`$BASE_MODEL_TAG\` step \`$BASE_MODEL_STEP\`"
append_report "- output_tag: \`$OUTPUT_TAG\`"
append_report "- sft_data: \`$SFT_DATA_PATH\`"
append_report "- include_smoltalk: \`$INCLUDE_SMOLTALK\`"
append_report "- train_steps: \`$TRAIN_STEPS\`"
append_report ""
append_report "Prompt/system/user tokens are kept fixed. Only assistant answer tokens are eligible for masking and reconstruction."
append_report ""

append_report "## Before SFT Samples"
append_report ""
run_python -m scripts.diffusion_sample_sweep \
  --source=diffusion \
  --model-tag="$BASE_MODEL_TAG" \
  --step="$BASE_MODEL_STEP" \
  --device-type="$DEVICE_TYPE" \
  --prompt="Explain masked diffusion language models in one paragraph." \
  --prompt="Write a tiny Python function that reverses a string." \
  --prompt="Give three practical tips for training a small language model." \
  --max-tokens="$SAMPLE_MAX_TOKENS" \
  --seed="$SAMPLE_SEED" \
  --output="$REPORT_FILE" \
  --append

run_python -m torch.distributed.run --standalone --nproc_per_node="$NPROC_PER_NODE" \
  -m scripts.diffusion_chat_sft -- \
  --run="$RUN_NAME" \
  --device-type="$DEVICE_TYPE" \
  --model-tag="$BASE_MODEL_TAG" \
  --model-step="$BASE_MODEL_STEP" \
  --data-jsonl="$SFT_DATA_PATH" \
  --num-iterations="$TRAIN_STEPS" \
  --max-seq-len="$MAX_SEQ_LEN" \
  --device-batch-size="$DEVICE_BATCH_SIZE" \
  --total-batch-size="$TOTAL_BATCH_SIZE" \
  --warmup-ratio="$WARMUP_RATIO" \
  --eval-every="$EVAL_EVERY" \
  --eval-batches="$EVAL_BATCHES" \
  --save-every="$SAVE_EVERY" \
  --output-tag="$OUTPUT_TAG" \
  "${smoltalk_args[@]}" \
  "${torch_args[@]}" 2>&1 | tee "$SFT_LOG"

append_report "## SFT Training Summary"
append_report ""
append_report '```text'
grep -E "SFT dataset rows|Step [0-9]+ \\| validation diffusion SFT loss|Peak memory usage|Total training time|Minimum validation diffusion SFT loss" "$SFT_LOG" >> "$REPORT_FILE" || true
append_report '```'
append_report ""

append_report "## After SFT Samples"
append_report ""
run_python -m scripts.diffusion_sample_sweep \
  --source=diffusion_sft \
  --model-tag="$OUTPUT_TAG" \
  --step="$TRAIN_STEPS" \
  --device-type="$DEVICE_TYPE" \
  --prompt="Explain masked diffusion language models in one paragraph." \
  --prompt="Write a tiny Python function that reverses a string." \
  --prompt="Give three practical tips for training a small language model." \
  --max-tokens="$SAMPLE_MAX_TOKENS" \
  --seed="$SAMPLE_SEED" \
  --output="$REPORT_FILE" \
  --append

append_report "- finished: $(date)"
append_report "- sft_checkpoints: \`$NANODIFFUSION_BASE_DIR/diffusion_sft_checkpoints/$OUTPUT_TAG\`"
append_report "- sft_log: \`$SFT_LOG\`"
echo "[diffusion_sft_a100] done: $REPORT_FILE"
