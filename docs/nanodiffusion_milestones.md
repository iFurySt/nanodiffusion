# NanoDiffusion Milestones

Date: 2026-05-20

This document turns the current NanoDiffusion prototype into a concrete
teaching-oriented roadmap. The goal is not SOTA. The goal is a nanochat-style
recipe that a reader can run on 8xA100 or 8xH100 in a few hours and understand
end to end.

## North Star

Build a reproducible masked diffusion language model pipeline:

```bash
bash runs/diffusion_speedrun_a100.sh
```

The script should produce:

- ClimbMix data and tokenizer artifacts.
- A base masked diffusion checkpoint.
- Validation diffusion loss.
- Fixed prompt samples at known checkpoints.
- A concise run report with model size, tokens, throughput, loss, samples, and
  command lines.

A follow-up script should produce an instruction-tuned checkpoint:

```bash
bash runs/diffusion_sft_a100.sh
```

The SFT script should produce:

- A diffusion SFT checkpoint.
- Fixed chat samples before and after SFT.
- A short note explaining that prompts are kept fixed and only assistant answer
  tokens are masked.

## References To Use

Use these projects as recipe references, not as code to copy wholesale:

- LLaDA: https://github.com/ML-GSAI/LLaDA
  - Main reference for masked diffusion pretraining and response-only SFT.
  - Relevant ideas: mask-probability sampling, prompt-fixed SFT, confidence
    remasking during generation.
- MDLM: https://github.com/kuleshov-group/mdlm
  - Main reference for masked diffusion configs and objective variants.
  - Relevant ideas: `mask_index = vocab_size`, continuous-time masked loss,
    sampler variants, OpenWebText-scale baselines.
- SEDD: https://github.com/louaaron/Score-Entropy-Discrete-Diffusion
  - Reference for stronger discrete diffusion objectives and evaluation style.
  - Keep as a v2 track after the simple masked baseline is stable.
- BD3-LMs: https://github.com/kuleshov-group/bd3lms
  - Reference for block diffusion and arbitrary-length generation.
  - Keep as a v2 track after fixed-length generation is understood.

## Current Baseline

Already implemented:

- `nanochat/diffusion.py`: mask id, masked batch creation, denoising loss,
  iterative sampler.
- `nanochat/gpt.py`: causal and bidirectional attention modes.
- `scripts/diffusion_base_train.py`: base masked diffusion pretraining.
- `scripts/diffusion_base_eval.py`: loss and sample evaluation.
- `scripts/diffusion_chat_sft.py`: answer-only diffusion SFT.
- `scripts/diffusion_chat_cli.py`: simple diffusion sampling CLI.
- `runs/diffusion_runcpu.sh`: tiny end-to-end smoke run.
- `runs/diffusion_speedrun.sh`: current 8xGPU base training entrypoint.

Known evidence:

- Local diffusion tests pass.
- A100 smoke runs pass.
- A 520M parameter, 8xA100, 500-step real-data run completed from 2 ClimbMix
  shards.
- The sample path works, but the 500-step model repeats heavily. That run is an
  engineering sanity check, not a quality baseline.
- `runs/diffusion_speedrun_a100.sh` completed a 10-shard, d20, seq 2048,
  5k-step run on 8xA100-80GB on 2026-05-20. It produced checkpoints every
  1000 steps, a markdown report, and a final validation diffusion loss of
  3.054810 after 167.64 minutes at roughly 260k tokens/sec.
- The 5k base samples are better than the 500-step sanity run but remain
  dominated by repeated short phrases and factual drift. This checkpoint is a
  reproducible engineering baseline, not the selected quality baseline.
- `runs/diffusion_sft_a100.sh` completed a 100-step 8xA100 smoke run from that
  base checkpoint after fixing optional SmolTalk imports and tiny-dataset DDP
  cursor wrapping. The smoke verified checkpoint loading, response-only
  masking, before/after sample reporting, and SFT checkpoint writing.
- The sampler now supports `--no-repeat-ngram-size`; a step-5000 A100 sample
  report showed that trigram blocking reduces exact loops on the same
  checkpoint, but does not solve the quality issue by itself.

## Milestone 1: Reproducible Base Speedrun

Purpose: create the first stable base model recipe.

Target:

- Hardware: 8xA100-80GB or 8xH100.
- Wall clock: 2-6 hours.
- Data: start with 10-20 ClimbMix shards.
- Model: keep `d20` as the first reference point unless memory/quality says
  otherwise.
- Sequence length: compare `1024` vs `2048`.
- Training length: start with `5k` steps; extend to `10k` only if samples are
  still improving.

Deliverables:

- Add `runs/diffusion_speedrun_a100.sh`.
- Script stages:
  - download data if missing
  - train tokenizer if missing
  - run base pretraining
  - evaluate validation diffusion loss
  - sample fixed prompts
  - write a concise report file
- Save checkpoints every `1000` or `2000` steps.
- Save fixed-prompt samples at every checkpoint.

Fixed prompts:

```text
The capital of France is
Once upon a time
In a shocking finding, scientists discovered
The meaning of life is
def fibonacci(n):
```

Acceptance gate:

- Script runs from a clean cache on A100/H100 without manual intervention.
- The final checkpoint loads with `scripts.diffusion_base_eval`.
- Validation loss trends down across checkpoints.
- Samples are visibly better than the 500-step sanity run, even if still weak.
- README documents the expected runtime and output paths.

## Milestone 2: Sampling Recipe

Purpose: reduce repetition and make base samples useful enough for teaching.

Experiments:

- Denoising steps:
  - generated length
  - half generated length
  - 2x generated length
- Sampling temperature:
  - `0.0`
  - `0.7`
  - `1.0`
- Top-k:
  - disabled
  - `50`
  - `200`
- Reveal/remask strategy:
  - current highest-confidence reveal
  - low-confidence remasking
  - fixed reveal schedule
  - block-wise generation as a simple BD3-inspired variant
- Repetition control:
  - suppress immediate repeated n-grams during sampling
  - penalize tokens already emitted in the generated span

Deliverables:

- Add CLI flags only for options that are useful and explainable.
- Add a small sampler comparison script or report mode.
- Update README with the default sampling recipe.

Acceptance gate:

- Same checkpoint produces less repetitive samples under the chosen default.
- The default remains simple enough to explain in the README.
- Tests cover prompt preservation, forbidden mask token output, and one sampler
  option that affects output.

## Milestone 3: Base Training Recipe Sweep

Purpose: find the smallest recipe that gives a credible teaching baseline.

Sweep dimensions:

- Data shards: `10`, `20`, `50` if time allows.
- Steps: `5k`, `10k`.
- Sequence length: `1024`, `2048`.
- Model size: current `d20`; optionally a smaller `d16` for faster iteration.
- Mask objective:
  - current uniform `t`
  - clamp high mask probabilities
  - compare loss weighting with and without `/ p_mask`

Record for every run:

- command
- commit
- data shards
- token count estimate
- model size
- throughput
- peak memory
- validation loss curve
- fixed-prompt samples

Acceptance gate:

- Pick one default base recipe.
- The selected recipe fits in a few hours on 8xA100/H100.
- The recipe is stable across at least two fresh launches or resume paths.

## Milestone 4: SFT Speedrun

Purpose: produce a chat-style diffusion model after base pretraining.

Target:

- Start only after base samples are no longer dominated by short repeated
  phrases.
- Keep prompt/system/user tokens fixed.
- Mask and train only assistant answer tokens.
- Use a small curated JSONL first, then add SmolTalk or a similar open dataset.

Deliverables:

- Add `runs/diffusion_sft_a100.sh`.
- Include:
  - base checkpoint path
  - SFT data path
  - SFT training
  - chat CLI samples
  - before/after sample report

Fixed chat prompts:

```text
Explain masked diffusion language models in one paragraph.
Write a tiny Python function that reverses a string.
Give three practical tips for training a small language model.
```

Acceptance gate:

- SFT checkpoint loads from `scripts.diffusion_chat_cli`.
- Response-only masking is verified by tests or a debug report.
- Chat samples show instruction following better than the base model.

## Milestone 5: Public Teaching Polish

Purpose: make the repo useful to outside readers.

Deliverables:

- README has a short end-to-end path:
  - CPU smoke
  - A100/H100 speedrun
  - sample
  - SFT
- README includes expected outputs from the selected baseline.
- Research doc links to the final recipe and explains what was borrowed from
  LLaDA/MDLM versus what was simplified.
- Add a troubleshooting section:
  - Hugging Face download mirror
  - CUDA memory
  - `torchrun` argument separator
  - where checkpoints are saved
  - how to resume
- Keep inherited nanochat attribution clear.

Acceptance gate:

- A new reader can run the CPU smoke without GPU access.
- A GPU user can run the A100/H100 speedrun with one command after dependency
  setup.
- All documented commands are checked before publishing.

## Things To Avoid For Now

- Do not chase SOTA metrics before the recipe is reproducible.
- Do not add SEDD or BD3 complexity to the first public baseline.
- Do not tune only by validation loss; always inspect fixed-prompt samples.
- Do not run SFT on a base checkpoint that only repeats phrases.
- Do not hide important defaults in ad hoc shell history. Put useful recipes in
  scripts and reports.

## Immediate Next Action

Use the completed 10-shard A100 run as the first reproducible engineering
baseline, then improve quality with the Milestone 2 and 3 sampler/training
sweeps before treating SFT outputs as model-quality evidence.

Then use that run to decide whether the next bottleneck is training length,
masking schedule, or sampler repetition.

Concrete next run: resume the 10-shard d20 seq-2048 baseline from step 5000 to
step 10000, keeping the same objective, to check whether the still-improving
loss and samples justify longer training before changing the masking objective.
