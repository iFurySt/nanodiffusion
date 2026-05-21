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

## Current Status

The engineering pipeline is reproducible: CPU smoke, 8xA100 base speedruns,
fixed-prompt sampling reports, and response-only SFT smoke have all run.

The quality baseline is still open. The best-tested base checkpoints still show
prompt-word repetition, factual drift, and unusable code continuations, so SFT
from those checkpoints should be treated as pipeline validation rather than a
recommended chat model.

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

Verified CPU smoke, 2026-05-22:

```bash
rm -rf /tmp/nanodiffusion-cpu-smoke
PYTHONPATH=. NANODIFFUSION_BASE_DIR=/tmp/nanodiffusion-cpu-smoke \
  uv run bash runs/diffusion_runcpu.sh
```

This run downloaded the two tiny data shards, trained the tokenizer, trained the
CPU diffusion checkpoint to step 20, saved
`/tmp/nanodiffusion-cpu-smoke/diffusion_checkpoints/diffusion_cpu/model_000020.pt`,
and completed a sample pass. The best validation diffusion loss in the smoke
run was `6.038887`.

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
MASK_EPS=0.7 MASK_PATTERN=suffix_span bash runs/diffusion_speedrun_a100.sh
LOSS_NORMALIZATION=eligible MASK_PATTERN=suffix_span bash runs/diffusion_speedrun_a100.sh
MASK_LOSS_REWEIGHT=0 bash runs/diffusion_speedrun_a100.sh
MASK_PATTERN=suffix bash runs/diffusion_speedrun_a100.sh
MASK_PATTERN=suffix_span SPAN_TOKENS=128 bash runs/diffusion_speedrun_a100.sh
MASK_PATTERN=suffix_all LOSS_NORMALIZATION=eligible bash runs/diffusion_speedrun_a100.sh
MASK_PATTERN=suffix_span_all SPAN_TOKENS=16 LOSS_NORMALIZATION=eligible MASK_LOSS_REWEIGHT=0 bash runs/diffusion_speedrun_a100.sh
MASK_PATTERN=suffix_span_mixed SPAN_TOKENS=64 LOSS_NORMALIZATION=eligible MASK_LOSS_REWEIGHT=0 bash runs/diffusion_speedrun_a100.sh
MASK_SAMPLING=antithetic bash runs/diffusion_speedrun_a100.sh
LOSS_OBJECTIVE=score_entropy MASK_MAX_PROB=0.999 MASK_SAMPLING=antithetic bash runs/diffusion_speedrun_a100.sh
LOSS_OBJECTIVE=score_entropy SCORE_PARAMETERIZATION=sigma_scaled MASK_MAX_PROB=0.999 MASK_SAMPLING=antithetic bash runs/diffusion_speedrun_a100.sh
```

The defaults keep the original simple LLaDA/MDLM-style objective: sampled mask
probability up to `1.0` and per-token loss divided by the row mask probability.
`MASK_PATTERN=suffix` keeps a random prefix visible and trains diffusion only on
the suffix, which is the next candidate for aligning base training with
fixed-prompt continuation. `MASK_PATTERN=suffix_span` narrows that objective to
a bounded continuation span and masks the future suffix without adding it to the
loss, which better matches fixed-length prompt sampling. Raising `MASK_EPS`
with `suffix_span` tests the fully masked continuation regime used at the start
of sampling. `LOSS_NORMALIZATION=eligible` keeps prompt-fixed objectives from
shrinking the gradient just because only a suffix or bounded span is trainable.
`MASK_PATTERN=suffix_all` masks the entire random suffix and trains all suffix
targets from the visible prefix only; it removes clean-suffix leakage but is a
much harder continuation objective.
`MASK_SAMPLING=antithetic` spreads mask probabilities across rows in each batch
instead of sampling every row independently.

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

A fresh 20-shard suffix run with the same model and sequence length completed
after that:

```text
model_tag: diffusion_a100_d20_s2048_5k_suffix_20s
data_shards: 20
training_time: 167.35m
minimum_validation_diffusion_loss: 1.668640
final_eval_loss: 1.791036
report: $NANODIFFUSION_BASE_DIR/report/diffusion_a100_d20_s2048_5k_suffix_20s-20260521-001927.md
```

It slightly improved the best validation loss, but fixed-prompt samples still
failed the same quality gate: phrase/list loops, factual drift, and no usable
code continuation. The selected quality baseline is still open.

A shorter sequence-length candidate improved validation loss further:

```text
model_tag: diffusion_a100_d20_s1024_5k_suffix_20s
data_shards: 20
max_seq_len: 1024
training_time: 157.12m
minimum_validation_diffusion_loss: 1.638211
final_eval_loss: 1.808493
report: $NANODIFFUSION_BASE_DIR/report/diffusion_a100_d20_s1024_5k_suffix_20s-20260521-033427.md
```

The lower validation loss did not translate into the fixed-prompt quality gate:
ordinary prompts still looped, and `def fibonacci(n):` mostly repeated malformed
function names. Resuming this checkpoint toward 10k was stopped at step 7000
after validation regressed (`1.736814 -> 1.951241 -> 1.934933 -> 1.897121 ->
1.816883`) and a step-7000 sample report still showed the same failure mode.

A short-prefix seq-1024 variant (`PREFIX_MIN_FRAC=0.0`,
`PREFIX_MAX_FRAC=0.25`) was stopped at step 1000. It was intended to align
training with very short fixed prompts, but validation stayed far worse
(`9.108521 -> 3.620450 -> 3.457720`) and step-1000 samples still repeated
prompt words and malformed function names.

A seq-256 suffix run was stopped at step 3000. It trained faster, but validation
remained worse than seq-1024 (`5.203685 -> 2.278308 -> 2.139555 -> 2.104049 ->
2.127015 -> 2.033860 -> 2.013864`) and the step-3000 sample report showed the
same loop failure, including `def fibonacci(n):` degenerating into repeated
parentheses and function-name fragments.

A bounded continuation-span objective was added after that:

```text
model_tag: diffusion_a100_d20_s1024_5k_suffix_span_20s
data_shards: 20
max_seq_len: 1024
mask_pattern: suffix_span
span_tokens: 128
training_time: 157.01m
minimum_validation_diffusion_loss: 0.473883
final_eval_loss: 0.466233
report: $NANODIFFUSION_BASE_DIR/report/diffusion_a100_d20_s1024_5k_suffix_span_20s-20260521-095859.md
```

The loss is not directly comparable with whole-suffix runs because only the
bounded span contributes targets. Samples still failed the quality gate:
`The capital of France is` looped around "capital", and `def fibonacci(n):`
remained non-code. This is a rejected recipe, not the selected quality baseline.

A high-mask variant of the same span objective was stopped early:

```text
model_tag: diffusion_a100_d20_s1024_5k_suffix_span_m070_20s
mask_pattern: suffix_span
mask_eps: 0.7
span_tokens: 128
stopped_at: step 1000
validation_loss: 1.300002 -> 0.786628 -> 0.737556
sample_report: $NANODIFFUSION_BASE_DIR/report/diffusion_a100_d20_s1024_5k_suffix_span_m070_20s-step1000-samples-20260521.md
```

This better matched the all-masked start of sampling, but it was worse on
validation and produced more severe fixed-prompt degeneration at step 1000, so
it was stopped instead of running to 5k.

An eligible-normalized bounded-span run was tested to avoid shrinking gradients
by the `span_tokens / max_seq_len` ratio:

```text
model_tag: diffusion_a100_d20_s1024_5k_suffix_span_elig_20s
mask_pattern: suffix_span
loss_normalization: eligible
span_tokens: 128
trained_to: step 1000
training_time: 31.18m
validation_loss: 10.471129 -> 4.889950 -> 4.247498
final_eval_loss: 4.195501
report: $NANODIFFUSION_BASE_DIR/report/diffusion_a100_d20_s1024_5k_suffix_span_elig_20s-20260521-133041.md
```

The larger loss is expected because it is now normalized by eligible tokens.
Samples at step 1000 were still dominated by loops, so it was not resumed to
5k. A separate `block_size=1` spot check on the best suffix and span checkpoints
also failed the fixed-prompt gate.

A 50-shard seq-1024 suffix run gave a small validation improvement but still
failed the fixed-prompt gate:

```text
model_tag: diffusion_a100_d20_s1024_5k_suffix_50s
data_shards: 50
max_seq_len: 1024
mask_pattern: suffix
training_time: 157.43m
minimum_validation_diffusion_loss: 1.632275
final_eval_loss: 1.804405
report: $NANODIFFUSION_BASE_DIR/report/diffusion_a100_d20_s1024_5k_suffix_50s-20260521-140725.md
```

The step-5000 validation loss was slightly better than the 20-shard seq-1024
suffix run (`1.632275` vs `1.638211`), but samples still looped and code prompts
remained unusable. Resuming toward 10k was stopped at step 5500 after validation
regressed to `1.951706`.

Two sampler/objective alignment checks were rejected after short runs:

- `--cfg-scale` adds classifier-free guidance, but spot checks on the best
  suffix checkpoints did not improve the fixed prompts; larger guidance often
  made samples less stable.
- `diffusion_a100_d20_s1024_5k_block4_20s` trained 4-token fully masked blocks
  to match `block_size=4` sampling. It was stopped at step 1000
  (`10.401340 -> 7.085275 -> 6.544755`) because samples were worse.

A capped-mask seq-1024 suffix run improved its own validation objective but did
not improve samples:

```text
model_tag: diffusion_a100_d20_s1024_5k_suffix_maxp070_20s
data_shards: 20
max_seq_len: 1024
mask_pattern: suffix
mask_max_prob: 0.7
training_time: 157.59m
minimum_validation_diffusion_loss: 1.224907
final_eval_loss: 1.253875
report: $NANODIFFUSION_BASE_DIR/report/diffusion_a100_d20_s1024_5k_suffix_maxp070_20s-20260521-175947.md
```

This objective is easier than the uncapped objective, so loss is not directly
comparable. Fixed-prompt samples still failed with factual drift and non-code
continuations.

A fixed left-to-right reveal schedule was also tested as a sampler-only change
on the seq-1024 suffix checkpoints:

```text
sample_reports:
  $NANODIFFUSION_BASE_DIR/report/diffusion_a100_d20_s1024_5k_suffix_20s-left-to-right-samples-20260521-205727.md
  $NANODIFFUSION_BASE_DIR/report/diffusion_a100_d20_s1024_5k_suffix_50s-left-to-right-samples-20260521-210014.md
  $NANODIFFUSION_BASE_DIR/report/diffusion_a100_d20_s1024_5k_suffix_maxp070_20s-left-to-right-samples-20260521-210145.md
```

The fixed schedule sometimes made prose more locally continuous than
highest-confidence reveal, but it still failed the fixed-prompt gate: factual
prompts drifted or repeated, and `def fibonacci(n):` did not produce usable
code. It is kept as an explainable sampling option, not as the selected default.

A fully masked suffix objective was added to test a broader continuation
training change:

```text
model_tag: diffusion_a100_d20_s1024_1k_suffix_all_20s
data_shards: 20
max_seq_len: 1024
mask_pattern: suffix_all
loss_normalization: eligible
trained_to: step 1000
training_time: 32.26m
final_eval_loss: 7.378942
report: $NANODIFFUSION_BASE_DIR/report/diffusion_a100_d20_s1024_1k_suffix_all_20s-20260521-210950.md
```

This removed future clean-suffix context during training, but the 1k pilot was
clearly worse than the rejected suffix/span candidates. Fixed-prompt samples
collapsed into France/French/Paris loops and character-level code degeneration,
so it should not be continued to 5k without another change.

A smaller d16 seq-1024 suffix pilot was also tested as a faster-iteration model
size check:

```text
model_tag: diffusion_a100_d16_s1024_1k_suffix_20s
data_shards: 20
depth: 16
max_seq_len: 1024
mask_pattern: suffix
trained_to: step 1000
training_time: 20.88m
minimum_validation_diffusion_loss: 2.019670
final_eval_loss: 1.905122
report: $NANODIFFUSION_BASE_DIR/report/diffusion_a100_d16_s1024_1k_suffix_20s-20260521-215254.md
```

The smaller model improved iteration speed but did not improve the fixed-prompt
gate. The France prompt drifted into comparison loops, and code prompts still
repeated malformed function names and fragments, so this pilot should not be
continued without another recipe change.

Training now excludes the reserved `[MASK]` id from the reconstruction softmax,
matching sampling where `[MASK]` is forbidden as an output. A 1k d20 seq-1024
suffix pilot tested this objective correction:

```text
model_tag: diffusion_a100_d20_s1024_1k_suffix_nomasklogit_20s
data_shards: 20
max_seq_len: 1024
mask_pattern: suffix
trained_to: step 1000
training_time: 32.55m
minimum_validation_diffusion_loss: 1.834602
final_eval_loss: 1.995015
report: $NANODIFFUSION_BASE_DIR/report/diffusion_a100_d20_s1024_1k_suffix_nomasklogit_20s-20260521-222757.md
```

This is the right training support, but the pilot still failed the fixed-prompt
gate: factual prompts drifted and repeated, and code prompts remained non-code.
It should not be continued to 5k by itself.

An antithetic mask-probability pilot was tested after the mask-logit correction:

```text
model_tag: diffusion_a100_d20_s1024_1k_suffix_antithetic_20s
data_shards: 20
max_seq_len: 1024
mask_pattern: suffix
mask_sampling: antithetic
trained_to: step 1000
training_time: 32.58m
minimum_validation_diffusion_loss: 1.810764
final_eval_loss: 1.840987
report: $NANODIFFUSION_BASE_DIR/report/diffusion_a100_d20_s1024_1k_suffix_antithetic_20s-20260521-231351.md
```

The validation loss was slightly better than the uniform mask-sampling pilot,
but fixed-prompt samples still failed with prompt-word loops and non-code
continuations. Keep antithetic sampling as a variance-reduction option, not as a
selected baseline.

An exact fully masked continuation-span pilot was tested to align training with
block-wise sampling without relying on `MASK_EPS=0.999` as an approximation:

```text
model_tag: diffusion_a100_d20_s1024_1k_suffix_span_all16_20s
data_shards: 20
max_seq_len: 1024
mask_pattern: suffix_span_all
span_tokens: 16
loss_normalization: eligible
mask_loss_reweight: 0
trained_to: step 1000
training_time: 32.44m
validation_loss_curve: 10.400656 -> 7.141670 -> 6.765229
final_eval_loss: 6.730374
report: $NANODIFFUSION_BASE_DIR/report/diffusion_a100_d20_s1024_1k_suffix_span_all16_20s-20260522-000814.md
```

This objective is explicit and better aligned with block-wise sampling, but the
pilot still failed the fixed-prompt gate with prompt-adjacent word loops and
non-code continuations. Do not continue it to 5k without a broader change.

A mixed continuation-span pilot trained half the rows with a fully masked span
and half with the ordinary no-future-leak span:

```text
model_tag: diffusion_a100_d20_s1024_1k_suffix_span_mixed64_20s
data_shards: 20
max_seq_len: 1024
mask_pattern: suffix_span_mixed
span_tokens: 64
loss_normalization: eligible
mask_loss_reweight: 0
trained_to: step 1000
training_time: 32.41m
validation_loss_curve: 7.670461 -> 5.141562 -> 4.825332
final_eval_loss: 4.672655
report: $NANODIFFUSION_BASE_DIR/report/diffusion_a100_d20_s1024_1k_suffix_span_mixed64_20s-20260522-004946.md
```

This was better than pure `suffix_span_all`, but fixed-prompt samples still
failed with prompt-word loops and non-code continuations. Treat it as useful
negative evidence, not a selected baseline.

A corrected full-objective control was run after mask-logit exclusion and
antithetic mask sampling were both available:

```text
model_tag: diffusion_a100_d20_s1024_1k_full_antithetic_20s
data_shards: 20
max_seq_len: 1024
mask_pattern: full
mask_sampling: antithetic
trained_to: step 1000
training_time: 32.57m
validation_loss_curve: 10.446225 -> 4.025616 -> 3.551752
final_eval_loss: 3.602521
report: $NANODIFFUSION_BASE_DIR/report/diffusion_a100_d20_s1024_1k_full_antithetic_20s-20260522-013001.md
```

The loss improved, but fixed-prompt samples still repeated prompt-adjacent words
and the code prompt did not produce usable code. This control is also not a
selected baseline.

An autoregressive control with the same d20, seq-1024, 20-shard data path and
global batch reached validation BPB `0.857841` after 1k steps in 17.21 minutes:

```text
model_tag: ar_d20_s1024_1k_20s_control
validation_bpb_curve: 3.171669 -> 0.943486 -> 0.857841
log: $NANODIFFUSION_BASE_DIR/logs/ar_d20_s1024_1k_20s_control-20260522-021154.train.log
```

The AR control still repeated and did not solve the code prompt, but it produced
much more language-like fixed-prompt continuations than the diffusion pilots
after the same 1k-step budget. This points the next work at the diffusion
objective/sampler rather than the shared data/tokenizer path alone.

Random remasking is available as a sampler comparison recipe:

```bash
python -m scripts.diffusion_base_eval \
  --model-tag=diffusion_a100_d20_s1024_5k_suffix_20s \
  --step=5000 \
  --eval=sample \
  --prompt="The capital of France is" \
  --temperature=0.8 \
  --top-k=50 \
  --no-repeat-ngram-size=3 \
  --remask-strategy=random
```

Spot checks on the suffix and corrected full checkpoints did not clear the
sample gate. Random remasking sometimes nudges factual prompts toward
Paris-like continuations, but it adds noise and does not fix code prompts.

A SEDD-inspired absorbing score-entropy objective is available:

```bash
LOSS_OBJECTIVE=score_entropy MASK_MAX_PROB=0.999 MASK_SAMPLING=antithetic \
  bash runs/diffusion_speedrun_a100.sh

LOSS_OBJECTIVE=score_entropy SCORE_PARAMETERIZATION=sigma_scaled \
  MASK_MAX_PROB=0.999 MASK_SAMPLING=antithetic \
  bash runs/diffusion_speedrun_a100.sh
```

A 1k d20 seq-1024 pilot ran stably but did not clear the sample gate:

```text
model_tag: diffusion_a100_d20_s1024_1k_score_entropy_full_20s
loss_objective: score_entropy
validation_loss_curve: 164823.796875 -> 6.875257 -> 4.694990
final_eval_loss: 4.742321
training_time: 32.78m
report: $NANODIFFUSION_BASE_DIR/report/diffusion_a100_d20_s1024_1k_score_entropy_full_20s-20260522-024848.md
```

Treat this as infrastructure for stronger discrete-diffusion experiments, not as
a selected baseline. The direct score-entropy objective still needs a matching
sampler or a better parameterization. `SCORE_PARAMETERIZATION=sigma_scaled`
adds the SEDD-style `sigma` scale correction to the loss path for the next pilot
while preserving `raw` as the default.

The sigma-scaled 1k pilot fixed the raw objective's huge initial loss scale, but
still did not clear the sample gate:

```text
model_tag: diffusion_a100_d20_s1024_1k_score_entropy_scaled_full_20s
loss_objective: score_entropy
score_parameterization: sigma_scaled
validation_loss_curve: 10.423655 -> 4.129064 -> 3.590117
final_eval_loss: 3.636929
training_time: 33.09m
report: $NANODIFFUSION_BASE_DIR/report/diffusion_a100_d20_s1024_1k_score_entropy_scaled_full_20s-20260522-033603.md
```

Fixed-prompt samples remained repetitive: the France prompt produced "capital of
..." variants instead of "Paris", and `def fibonacci(n):` produced number or
topic lists rather than executable code. This makes sigma scaling a useful
stability improvement, not a quality baseline.

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
  --block-size=4 \
  --cfg-scale=1.5 \
  --reveal-strategy=confidence
```

For the current baseline, `--no-repeat-ngram-size=3 --block-size=4` is the
clearest sampling default: it prevents exact repeated trigrams and generates a
few tokens at a time, which is less repetitive than filling the whole answer
window at once. It still does not fix weak base-model knowledge or planning by
itself. `--cfg-scale` adds classifier-free guidance by comparing normal prompt
conditioning against a copy where the prompt tokens are masked.
`--reveal-strategy=left_to_right` uses a fixed reveal schedule instead of
highest-confidence reveal; it is useful for comparison but has not cleared the
quality gate on the current checkpoints.

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
