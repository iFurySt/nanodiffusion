# NanoDiffusion

NanoDiffusion is a small, readable fork of nanochat for training masked
diffusion Transformer language models.

The goal is the same spirit as nanochat: keep the codebase understandable, keep
the full training path runnable by normal people, and make one central idea easy
to inspect. Here the central idea is:

> Instead of predicting the next token from the left context, mask some tokens
> and train a bidirectional Transformer to reconstruct them.

This repo is early, but it already has the base model path:

- the original nanochat dataset and tokenizer pipeline
- a GPT-style Transformer that can run bidirectional attention
- one extra `[MASK]` id outside the tokenizer vocabulary
- LLaDA/MDLM-style masked denoising loss
- fixed-length iterative denoising sampling
- base training, evaluation, checkpointing, and speedrun entrypoints

The implementation is intentionally close to nanochat so that the difference
between autoregressive language modeling and diffusion language modeling stays
visible in the code.

## How It Works

Autoregressive GPT training learns:

```text
p(x_i | x_<i)
```

Generation appends one token at a time.

Masked diffusion training starts with a clean sequence:

```text
The cat sat on the mat
```

Then samples a noise level and replaces some tokens with `[MASK]`:

```text
The [MASK] sat [MASK] the mat
```

The model sees both left and right context and predicts only the masked original
tokens. At generation time, it starts from masks and fills them over multiple
rounds, keeping high-confidence predictions and revisiting the rest.

## Setup

NanoDiffusion uses [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra gpu
# or CPU/MPS:
uv sync --extra cpu
```

For development:

```bash
uv sync --extra cpu --group dev
```

By default intermediate files live in `~/.cache/nanodiffusion`. You can override
this with:

```bash
export NANODIFFUSION_BASE_DIR=/path/to/cache
```

`NANOCHAT_BASE_DIR` is still accepted for compatibility with inherited scripts.

## Data And Tokenizer

NanoDiffusion reuses nanochat's dataset/tokenizer path:

```bash
python -m nanochat.dataset -n 10
python -m scripts.tok_train --max-chars=2000000000
```

The tokenizer is not retrained to include `[MASK]`. Instead:

```text
mask_token_id = tokenizer.get_vocab_size()
model_vocab_size = tokenizer.get_vocab_size() + 1
```

The extra id exists only inside the model and diffusion scripts.

## Train

Small CPU smoke run, after data/tokenizer are prepared:

```bash
python -m scripts.diffusion_base_train \
  --device-type=cpu \
  --depth=2 \
  --aspect-ratio=16 \
  --head-dim=16 \
  --max-seq-len=64 \
  --device-batch-size=2 \
  --total-batch-size=128 \
  --num-iterations=5 \
  --eval-every=-1
```

Or run the tiny end-to-end CPU script:

```bash
bash runs/diffusion_runcpu.sh
```

8xGPU reference entrypoint for the first reproducible A100/H100 baseline:

```bash
bash runs/diffusion_speedrun_a100.sh
```

By default this downloads 10 ClimbMix shards, trains a 32k tokenizer if needed,
runs a `d20` masked diffusion model for 5k steps at sequence length 2048, saves
checkpoints every 1000 steps, and writes a markdown report under:

```text
$NANODIFFUSION_BASE_DIR/report/
```

The older `runs/diffusion_speedrun.sh` is kept as a minimal base-training
entrypoint. The A100/H100 script is the teaching recipe that records commands,
losses, throughput, memory, and fixed-prompt samples.

Observed 8xA100-80GB reference run, 2026-05-20:

```text
model_tag: diffusion_a100_d20_s2048_5k
data_shards: 10
parameters: 897,516,786
training_time: 167.64m
throughput: ~260k tokens/sec
peak_memory: 64.3GiB per rank during training
minimum_validation_diffusion_loss: 3.054810
report: $NANODIFFUSION_BASE_DIR/report/diffusion_a100_d20_s2048_5k-*.md
```

This is a reproducible engineering baseline, not a quality baseline yet. The
fixed-prompt samples improve over the 500-step sanity run but still show short
phrase repetition, so treat SFT from this checkpoint as a pipeline smoke test
unless you extend or improve the base recipe.

Useful sweep overrides for the next base runs:

```bash
MASK_MAX_PROB=0.7 bash runs/diffusion_speedrun_a100.sh
MASK_LOSS_REWEIGHT=0 bash runs/diffusion_speedrun_a100.sh
MASK_PATTERN=suffix bash runs/diffusion_speedrun_a100.sh
```

The defaults keep the original simple LLaDA/MDLM-style objective: sampled mask
probability up to `1.0` and per-token loss divided by the row mask probability.
`MASK_PATTERN=suffix` keeps a random prefix visible and trains diffusion only on
the suffix, which is the next candidate for aligning base training with
fixed-prompt continuation.

Additional A100 sweeps on 2026-05-20 showed that simply extending this 10-shard
baseline to 10k steps, or retraining with `MASK_LOSS_REWEIGHT=0`, did not clear
the repetition quality gate. The no-reweight run was stopped at step 3000 after
validation stopped improving; step 2000 had the lowest loss, but fixed-prompt
samples were still dominated by short loops.

A suffix-objective run (`MASK_PATTERN=suffix`, random visible prefix between
25% and 75% of the sequence) reached a much lower validation loss after resuming
from 2k to 5k steps:

```text
model_tag: diffusion_a100_d20_s2048_2k_suffix
run_name: diffusion_a100_d20_s2048_2k_suffix_resume5k
training_time: 167.58m for the 2k->5k resume
minimum_validation_diffusion_loss: 1.687262
final_eval_loss: 1.796170
report: $NANODIFFUSION_BASE_DIR/report/diffusion_a100_d20_s2048_2k_suffix-20260520-222324.md
```

This is useful evidence that the objective better matches fixed-prompt
continuation, but it is still not the selected quality baseline: samples remain
dominated by topical loops, factual drift, and code-prompt failures. Do not use
this checkpoint as the quality SFT base.

## Evaluate And Sample

Evaluate validation diffusion loss and print one sample:

```bash
python -m scripts.diffusion_base_eval \
  --model-tag=diffusion_d20 \
  --eval=loss,sample \
  --prompt="The capital of France is" \
  --max-tokens=32 \
  --temperature=0.8 \
  --top-k=50 \
  --repeat-penalty=0.5 \
  --no-repeat-ngram-size=3 \
  --block-size=4
```

For the current baseline, `--no-repeat-ngram-size=3 --block-size=4` is the
clearest sampling default: it prevents exact repeated trigrams and generates a
few tokens at a time, which is less repetitive than filling the whole answer
window at once. It still does not fix weak base-model knowledge or planning by
itself.

Sampling is fixed-length. Prompt tokens stay fixed; the remaining positions start
as `[MASK]` and are filled by iterative denoising.

For a small report that compares a few sampling recipes on the fixed prompts:

```bash
python -m scripts.diffusion_sample_sweep \
  --model-tag=diffusion_d20 \
  --step=5000 \
  --output=$NANODIFFUSION_BASE_DIR/report/diffusion_samples.md
```

For a small interactive loop around the same sampler:

```bash
python -m scripts.diffusion_chat_cli --model-tag=diffusion_d20
```

## Supervised Fine-Tuning

Diffusion SFT uses the same conversation rendering as nanochat, but the loss is
different: user/system/prompt tokens stay fixed, and only assistant-answer tokens
are eligible for masking and reconstruction.

JSONL format:

```jsonl
[{"role":"user","content":"Say hello"},{"role":"assistant","content":"Hello from NanoDiffusion."}]
```

Run:

```bash
python -m scripts.diffusion_chat_sft \
  --model-tag=diffusion_d20 \
  --data-jsonl=$NANODIFFUSION_BASE_DIR/identity_conversations.jsonl \
  --output-tag=diffusion_d20_sft
```

Then sample from the SFT checkpoint:

```bash
python -m scripts.diffusion_chat_cli \
  --source=diffusion_sft \
  --model-tag=diffusion_d20_sft
```

For the A100/H100 recipe after a base speedrun:

```bash
bash runs/diffusion_sft_a100.sh
```

The script writes a small curated JSONL if none is provided, samples fixed chat
prompts before SFT, trains response-only diffusion SFT, then samples the same
prompts from the SFT checkpoint. Use `SFT_DATA_PATH=/path/to/data.jsonl` or
`INCLUDE_SMOLTALK=1` to change the data mix.

## Important Files

```text
nanochat/gpt.py                  Transformer with causal/bidirectional modes
nanochat/diffusion.py            Masking, denoising loss, diffusion sampler
nanochat/flash_attention.py      FA3/SDPA attention wrapper
scripts/diffusion_base_train.py  Base masked diffusion pretraining
scripts/diffusion_base_eval.py   Validation loss and sampling
scripts/diffusion_sample_sweep.py Fixed-prompt sampler comparison report
scripts/diffusion_chat_sft.py    Prompt-fixed, answer-only masked SFT
scripts/diffusion_chat_cli.py    Minimal interactive diffusion sampler
runs/diffusion_speedrun_a100.sh  8xA100/H100 baseline recipe with reports
runs/diffusion_sft_a100.sh       Response-only diffusion SFT recipe
runs/diffusion_speedrun.sh       Minimal 8xGPU base training entrypoint
runs/diffusion_runcpu.sh         Tiny CPU/MPS learning run
docs/diffusion_language_model_research.md
docs/nanodiffusion_milestones.md
```

Inherited autoregressive nanochat scripts are still present while the fork is
being carved into a dedicated diffusion LM project.

## Research Direction

The first implementation track follows masked discrete diffusion, closest in
spirit to:

- LLaDA: https://arxiv.org/abs/2502.09992
- MDLM: https://arxiv.org/abs/2406.07524
- SEDD: https://arxiv.org/abs/2310.16834
- BD3-LM: https://arxiv.org/abs/2503.09573

See [docs/diffusion_language_model_research.md](docs/diffusion_language_model_research.md)
for the initial project notes and implementation plan.

## Troubleshooting

Hugging Face downloads:

```bash
export NANOCHAT_DATASET_BASE_URL=https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main
```

If your network needs a mirror, set `NANOCHAT_DATASET_BASE_URL` to a compatible
mirror before running `nanochat.dataset`.

CUDA memory:

- Reduce `DEVICE_BATCH_SIZE` first.
- Then reduce `MAX_SEQ_LEN` from `2048` to `1024`.
- Keep `TOTAL_BATCH_SIZE` divisible by `DEVICE_BATCH_SIZE * MAX_SEQ_LEN * NPROC_PER_NODE`.

`torchrun` arguments:

- Put training-script flags after `--` when invoking a module through
  `torchrun -m`, for example:

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.diffusion_base_train -- --run=my_run
```

Checkpoint paths:

```text
$NANODIFFUSION_BASE_DIR/diffusion_checkpoints/<model-tag>/
$NANODIFFUSION_BASE_DIR/diffusion_sft_checkpoints/<model-tag>/
```

Resume base training:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m scripts.diffusion_base_train -- \
  --model-tag=diffusion_a100_d20_s2048_5k \
  --resume-from-step=1000
```

## Attribution

NanoDiffusion is forked from Andrej Karpathy's
[nanochat](https://github.com/karpathy/nanochat). The dataset pipeline,
tokenizer, optimizer, reporting utilities, and much of the Transformer training
infrastructure come from nanochat.

## License

MIT
