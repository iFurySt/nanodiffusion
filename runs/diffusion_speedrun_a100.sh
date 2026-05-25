#!/usr/bin/env bash
set -euo pipefail

# Reproducible 8xA100/H100 masked diffusion base run.
# Defaults target the first public NanoDiffusion baseline recipe. Override any
# uppercase variable from the shell to run a smaller smoke or a longer baseline.

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NANODIFFUSION_BASE_DIR="${NANODIFFUSION_BASE_DIR:-$HOME/.cache/nanodiffusion-a100}"
export HF_HOME="${HF_HOME:-$NANODIFFUSION_BASE_DIR/huggingface}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$NANODIFFUSION_BASE_DIR/uv-cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$NANODIFFUSION_BASE_DIR/pip-cache}"

PYTHON_BIN="${PYTHON_BIN:-}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
DATA_SHARDS="${DATA_SHARDS:-10}"
TOKENIZER_MAX_CHARS="${TOKENIZER_MAX_CHARS:-2000000000}"
VOCAB_SIZE="${VOCAB_SIZE:-32768}"
MODEL_TAG="${MODEL_TAG:-diffusion_a100_d20_s2048_5k}"
RUN_NAME="${RUN_NAME:-$MODEL_TAG}"
DEPTH="${DEPTH:-20}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
ATTENTION_MODE="${ATTENTION_MODE:-bidirectional}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-16}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-524288}"
TRAIN_STEPS="${TRAIN_STEPS:-5000}"
WARMUP_STEPS="${WARMUP_STEPS:-100}"
EMBEDDING_LR="${EMBEDDING_LR:-0.3}"
UNEMBEDDING_LR="${UNEMBEDDING_LR:-0.008}"
MATRIX_LR="${MATRIX_LR:-0.02}"
SCALAR_LR="${SCALAR_LR:-0.5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.28}"
MASK_EPS="${MASK_EPS:-1e-3}"
MASK_MAX_PROB="${MASK_MAX_PROB:-1.0}"
MASK_LOSS_REWEIGHT="${MASK_LOSS_REWEIGHT:-1}"
MASK_PATTERN="${MASK_PATTERN:-full}"
PREFIX_MIN_FRAC="${PREFIX_MIN_FRAC:-0.25}"
PREFIX_MAX_FRAC="${PREFIX_MAX_FRAC:-0.75}"
SPAN_TOKENS="${SPAN_TOKENS:-128}"
LOSS_NORMALIZATION="${LOSS_NORMALIZATION:-all}"
MASK_SAMPLING="${MASK_SAMPLING:-uniform}"
LOSS_OBJECTIVE="${LOSS_OBJECTIVE:-cross_entropy}"
SCORE_PARAMETERIZATION="${SCORE_PARAMETERIZATION:-raw}"
DIFFUSION_SIGMA_CONDITIONING="${DIFFUSION_SIGMA_CONDITIONING:-0}"
DIFFUSION_SIGMA_LAYER_CONDITIONING="${DIFFUSION_SIGMA_LAYER_CONDITIONING:-0}"
DIFFUSION_SIGMA_ADALN_CONDITIONING="${DIFFUSION_SIGMA_ADALN_CONDITIONING:-0}"
DIFFUSION_SIGMA_EMBEDDING="${DIFFUSION_SIGMA_EMBEDDING:-scalar}"
DIFFUSION_SIGMA_EMBEDDING_DIM="${DIFFUSION_SIGMA_EMBEDDING_DIM:-256}"
INIT_FROM_BASE_MODEL_TAG="${INIT_FROM_BASE_MODEL_TAG:-}"
INIT_FROM_BASE_STEP="${INIT_FROM_BASE_STEP:--1}"
AR_TEACHER_MODEL_TAG="${AR_TEACHER_MODEL_TAG:-}"
AR_TEACHER_STEP="${AR_TEACHER_STEP:--1}"
AR_TEACHER_KL_WEIGHT="${AR_TEACHER_KL_WEIGHT:-0.0}"
AR_TEACHER_TEMPERATURE="${AR_TEACHER_TEMPERATURE:-1.0}"
AR_ROLLOUT_TOKENS="${AR_ROLLOUT_TOKENS:-0}"
AR_ROLLOUT_TEMPERATURE="${AR_ROLLOUT_TEMPERATURE:-0.8}"
AR_ROLLOUT_TOP_K="${AR_ROLLOUT_TOP_K:-50}"
EVAL_EVERY="${EVAL_EVERY:-500}"
EVAL_BATCHES="${EVAL_BATCHES:-20}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
SAMPLE_MAX_TOKENS="${SAMPLE_MAX_TOKENS:-64}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
SAMPLE_NO_REPEAT_NGRAM_SIZE="${SAMPLE_NO_REPEAT_NGRAM_SIZE:-3}"
SAMPLE_BLOCK_SIZE="${SAMPLE_BLOCK_SIZE:-4}"
SAMPLE_REVEAL_STRATEGY="${SAMPLE_REVEAL_STRATEGY:-confidence}"
SAMPLE_CFG_SCALE="${SAMPLE_CFG_SCALE:-0.0}"
SAMPLE_REMASK_LOW_CONFIDENCE="${SAMPLE_REMASK_LOW_CONFIDENCE:-0}"
SAMPLE_REMASK_STRATEGY="${SAMPLE_REMASK_STRATEGY:-none}"
SAMPLE_SAMPLER="${SAMPLE_SAMPLER:-iterative}"
RESUME_FROM_STEP="${RESUME_FROM_STEP:--1}"
COMPILE="${COMPILE:-0}"

mkdir -p "$NANODIFFUSION_BASE_DIR/logs" "$NANODIFFUSION_BASE_DIR/report"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
REPORT_FILE="$NANODIFFUSION_BASE_DIR/report/${MODEL_TAG}-${TIMESTAMP}.md"
TRAIN_LOG="$NANODIFFUSION_BASE_DIR/logs/${MODEL_TAG}-${TIMESTAMP}.train.log"
EVAL_LOG="$NANODIFFUSION_BASE_DIR/logs/${MODEL_TAG}-${TIMESTAMP}.eval.log"

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

torch_args=()
if [ "$COMPILE" = "1" ]; then
  torch_args+=(--compile)
fi
if [ "$RESUME_FROM_STEP" != "-1" ]; then
  torch_args+=(--resume-from-step="$RESUME_FROM_STEP")
fi
if [ "$MASK_LOSS_REWEIGHT" = "0" ]; then
  torch_args+=(--no-mask-loss-reweight)
fi
if [ "$DIFFUSION_SIGMA_CONDITIONING" = "1" ]; then
  torch_args+=(--diffusion-sigma-conditioning)
fi
if [ "$DIFFUSION_SIGMA_LAYER_CONDITIONING" = "1" ]; then
  torch_args+=(--diffusion-sigma-layer-conditioning)
fi
if [ "$DIFFUSION_SIGMA_ADALN_CONDITIONING" = "1" ]; then
  torch_args+=(--diffusion-sigma-adaln-conditioning)
fi
if [ -n "$INIT_FROM_BASE_MODEL_TAG" ]; then
  torch_args+=(--init-from-base-model-tag="$INIT_FROM_BASE_MODEL_TAG")
  torch_args+=(--init-from-base-step="$INIT_FROM_BASE_STEP")
fi
if [ -n "$AR_TEACHER_MODEL_TAG" ]; then
  torch_args+=(--ar-teacher-model-tag="$AR_TEACHER_MODEL_TAG")
  torch_args+=(--ar-teacher-step="$AR_TEACHER_STEP")
  torch_args+=(--ar-teacher-kl-weight="$AR_TEACHER_KL_WEIGHT")
  torch_args+=(--ar-teacher-temperature="$AR_TEACHER_TEMPERATURE")
fi
if [ "$AR_ROLLOUT_TOKENS" != "0" ]; then
  torch_args+=(--ar-rollout-tokens="$AR_ROLLOUT_TOKENS")
  torch_args+=(--ar-rollout-temperature="$AR_ROLLOUT_TEMPERATURE")
  torch_args+=(--ar-rollout-top-k="$AR_ROLLOUT_TOP_K")
fi

eval_args=(--mask-eps="$MASK_EPS")
eval_args+=(--mask-max-prob="$MASK_MAX_PROB")
eval_args+=(--mask-pattern="$MASK_PATTERN")
eval_args+=(--prefix-min-frac="$PREFIX_MIN_FRAC")
eval_args+=(--prefix-max-frac="$PREFIX_MAX_FRAC")
eval_args+=(--span-tokens="$SPAN_TOKENS")
eval_args+=(--loss-normalization="$LOSS_NORMALIZATION")
eval_args+=(--mask-sampling="$MASK_SAMPLING")
eval_args+=(--loss-objective="$LOSS_OBJECTIVE")
eval_args+=(--score-parameterization="$SCORE_PARAMETERIZATION")
if [ "$MASK_LOSS_REWEIGHT" = "0" ]; then
  eval_args+=(--no-mask-loss-reweight)
fi
if [ "$SAMPLE_REMASK_LOW_CONFIDENCE" = "1" ]; then
  eval_args+=(--remask-low-confidence)
fi
eval_args+=(--remask-strategy="$SAMPLE_REMASK_STRATEGY")
eval_args+=(--sampler="$SAMPLE_SAMPLER")

commit="$(git rev-parse HEAD 2>/dev/null || cat .sync/source_commit 2>/dev/null || echo unknown)"
append_report "# NanoDiffusion A100 Speedrun"
append_report ""
append_report "- started: $(date)"
append_report "- commit: \`$commit\`"
append_report "- base_dir: \`$NANODIFFUSION_BASE_DIR\`"
append_report "- model_tag: \`$MODEL_TAG\`"
append_report "- data_shards: \`$DATA_SHARDS\`"
append_report "- vocab_size: \`$VOCAB_SIZE\`"
append_report "- depth: \`$DEPTH\`"
append_report "- max_seq_len: \`$MAX_SEQ_LEN\`"
append_report "- attention_mode: \`$ATTENTION_MODE\`"
append_report "- train_steps: \`$TRAIN_STEPS\`"
append_report "- resume_from_step: \`$RESUME_FROM_STEP\`"
append_report "- embedding_lr: \`$EMBEDDING_LR\`"
append_report "- unembedding_lr: \`$UNEMBEDDING_LR\`"
append_report "- matrix_lr: \`$MATRIX_LR\`"
append_report "- scalar_lr: \`$SCALAR_LR\`"
append_report "- weight_decay: \`$WEIGHT_DECAY\`"
append_report "- mask_eps: \`$MASK_EPS\`"
append_report "- mask_max_prob: \`$MASK_MAX_PROB\`"
append_report "- mask_loss_reweight: \`$MASK_LOSS_REWEIGHT\`"
append_report "- mask_pattern: \`$MASK_PATTERN\`"
append_report "- prefix_min_frac: \`$PREFIX_MIN_FRAC\`"
append_report "- prefix_max_frac: \`$PREFIX_MAX_FRAC\`"
append_report "- span_tokens: \`$SPAN_TOKENS\`"
append_report "- loss_normalization: \`$LOSS_NORMALIZATION\`"
append_report "- mask_sampling: \`$MASK_SAMPLING\`"
append_report "- loss_objective: \`$LOSS_OBJECTIVE\`"
append_report "- score_parameterization: \`$SCORE_PARAMETERIZATION\`"
append_report "- diffusion_sigma_conditioning: \`$DIFFUSION_SIGMA_CONDITIONING\`"
append_report "- diffusion_sigma_layer_conditioning: \`$DIFFUSION_SIGMA_LAYER_CONDITIONING\`"
append_report "- diffusion_sigma_adaln_conditioning: \`$DIFFUSION_SIGMA_ADALN_CONDITIONING\`"
append_report "- diffusion_sigma_embedding: \`$DIFFUSION_SIGMA_EMBEDDING\`"
append_report "- diffusion_sigma_embedding_dim: \`$DIFFUSION_SIGMA_EMBEDDING_DIM\`"
append_report "- init_from_base_model_tag: \`$INIT_FROM_BASE_MODEL_TAG\`"
append_report "- init_from_base_step: \`$INIT_FROM_BASE_STEP\`"
append_report "- ar_teacher_model_tag: \`$AR_TEACHER_MODEL_TAG\`"
append_report "- ar_teacher_step: \`$AR_TEACHER_STEP\`"
append_report "- ar_teacher_kl_weight: \`$AR_TEACHER_KL_WEIGHT\`"
append_report "- ar_teacher_temperature: \`$AR_TEACHER_TEMPERATURE\`"
append_report "- ar_rollout_tokens: \`$AR_ROLLOUT_TOKENS\`"
append_report "- ar_rollout_temperature: \`$AR_ROLLOUT_TEMPERATURE\`"
append_report "- ar_rollout_top_k: \`$AR_ROLLOUT_TOP_K\`"
append_report "- sample_remask_low_confidence: \`$SAMPLE_REMASK_LOW_CONFIDENCE\`"
append_report "- sample_remask_strategy: \`$SAMPLE_REMASK_STRATEGY\`"
append_report "- sample_sampler: \`$SAMPLE_SAMPLER\`"
append_report "- sample_block_size: \`$SAMPLE_BLOCK_SIZE\`"
append_report "- sample_reveal_strategy: \`$SAMPLE_REVEAL_STRATEGY\`"
append_report "- sample_cfg_scale: \`$SAMPLE_CFG_SCALE\`"
append_report "- total_batch_size: \`$TOTAL_BATCH_SIZE\`"
append_report "- device_batch_size: \`$DEVICE_BATCH_SIZE\`"
append_report "- nproc_per_node: \`$NPROC_PER_NODE\`"
append_report "- estimated_training_tokens: \`$((TRAIN_STEPS * TOTAL_BATCH_SIZE))\`"
append_report ""
append_report "## Commands"
append_report ""
append_report '```bash'
append_report "python -m nanochat.dataset -n $DATA_SHARDS"
append_report "python -m scripts.tok_train --max-chars=$TOKENIZER_MAX_CHARS --vocab-size=$VOCAB_SIZE"
append_report "python -m torch.distributed.run --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.diffusion_base_train -- ..."
append_report '```'
append_report ""

echo "[diffusion_speedrun_a100] report: $REPORT_FILE"
echo "[diffusion_speedrun_a100] train log: $TRAIN_LOG"

run_python -m nanochat.dataset -n "$DATA_SHARDS"

if [ ! -f "$NANODIFFUSION_BASE_DIR/tokenizer/tokenizer.pkl" ] || [ "${FORCE_TOKENIZER:-0}" = "1" ]; then
  run_python -m scripts.tok_train --max-chars="$TOKENIZER_MAX_CHARS" --vocab-size="$VOCAB_SIZE"
else
  echo "[diffusion_speedrun_a100] tokenizer exists; set FORCE_TOKENIZER=1 to retrain"
fi

run_python -m nanochat.report reset

run_python -m torch.distributed.run --standalone --nproc_per_node="$NPROC_PER_NODE" \
  -m scripts.diffusion_base_train -- \
  --run="$RUN_NAME" \
  --depth="$DEPTH" \
  --max-seq-len="$MAX_SEQ_LEN" \
  --attention-mode="$ATTENTION_MODE" \
  --device-batch-size="$DEVICE_BATCH_SIZE" \
  --total-batch-size="$TOTAL_BATCH_SIZE" \
  --num-iterations="$TRAIN_STEPS" \
  --warmup-steps="$WARMUP_STEPS" \
  --embedding-lr="$EMBEDDING_LR" \
  --unembedding-lr="$UNEMBEDDING_LR" \
  --matrix-lr="$MATRIX_LR" \
  --scalar-lr="$SCALAR_LR" \
  --weight-decay="$WEIGHT_DECAY" \
  --mask-eps="$MASK_EPS" \
  --mask-max-prob="$MASK_MAX_PROB" \
  --mask-pattern="$MASK_PATTERN" \
  --prefix-min-frac="$PREFIX_MIN_FRAC" \
  --prefix-max-frac="$PREFIX_MAX_FRAC" \
  --span-tokens="$SPAN_TOKENS" \
  --loss-normalization="$LOSS_NORMALIZATION" \
  --mask-sampling="$MASK_SAMPLING" \
  --loss-objective="$LOSS_OBJECTIVE" \
  --score-parameterization="$SCORE_PARAMETERIZATION" \
  --diffusion-sigma-embedding="$DIFFUSION_SIGMA_EMBEDDING" \
  --diffusion-sigma-embedding-dim="$DIFFUSION_SIGMA_EMBEDDING_DIM" \
  --eval-every="$EVAL_EVERY" \
  --eval-batches="$EVAL_BATCHES" \
  --save-every="$SAVE_EVERY" \
  --model-tag="$MODEL_TAG" \
  "${torch_args[@]}" 2>&1 | tee "$TRAIN_LOG"

append_report "## Training Summary"
append_report ""
append_report '```text'
grep -E "Total parameters|Training iterations|Step [0-9]+ \\| validation diffusion loss|Peak memory usage|Total training time|Minimum validation diffusion loss" "$TRAIN_LOG" >> "$REPORT_FILE" || true
append_report '```'
append_report ""

append_report "## Fixed-Prompt Samples"
append_report ""
for step in $(seq "$SAVE_EVERY" "$SAVE_EVERY" "$TRAIN_STEPS"); do
  if [ -f "$NANODIFFUSION_BASE_DIR/diffusion_checkpoints/$MODEL_TAG/model_$(printf "%06d" "$step").pt" ]; then
    run_python -m scripts.diffusion_sample_sweep \
      --model-tag="$MODEL_TAG" \
      --step="$step" \
      --device-type=cuda \
      --max-tokens="$SAMPLE_MAX_TOKENS" \
      --seed="$SAMPLE_SEED" \
      --score-parameterization="$SCORE_PARAMETERIZATION" \
      --mask-eps="$MASK_EPS" \
      --mask-max-prob="$MASK_MAX_PROB" \
      --output="$REPORT_FILE" \
      --append
  fi
done

append_report "## Final Evaluation"
append_report ""
append_report '```text'
run_python -m scripts.diffusion_base_eval \
  --device-type=cuda \
  --model-tag="$MODEL_TAG" \
  --step="$TRAIN_STEPS" \
  --eval=loss,sample \
  --eval-batches="$EVAL_BATCHES" \
  --prompt="The capital of France is" \
  --max-tokens="$SAMPLE_MAX_TOKENS" \
  --temperature=0.8 \
  --top-k=50 \
  "${eval_args[@]}" \
  --repeat-penalty=0.5 \
  --no-repeat-ngram-size="$SAMPLE_NO_REPEAT_NGRAM_SIZE" \
  --block-size="$SAMPLE_BLOCK_SIZE" \
  --reveal-strategy="$SAMPLE_REVEAL_STRATEGY" \
  --cfg-scale="$SAMPLE_CFG_SCALE" 2>&1 | tee "$EVAL_LOG" | tee -a "$REPORT_FILE"
append_report '```'
append_report ""
append_report "- finished: $(date)"
append_report "- checkpoints: \`$NANODIFFUSION_BASE_DIR/diffusion_checkpoints/$MODEL_TAG\`"
append_report "- train_log: \`$TRAIN_LOG\`"
append_report "- eval_log: \`$EVAL_LOG\`"

run_python -m nanochat.report generate
echo "[diffusion_speedrun_a100] done: $REPORT_FILE"
