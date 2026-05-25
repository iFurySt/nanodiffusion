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
- A clean CPU smoke completed on 2026-05-22 with:
  `PYTHONPATH=. NANODIFFUSION_BASE_DIR=/tmp/nanodiffusion-cpu-smoke uv run bash runs/diffusion_runcpu.sh`.
  It downloaded the two tiny data shards, trained the tokenizer, trained a tiny
  CPU checkpoint to step 20, saved
  `/tmp/nanodiffusion-cpu-smoke/diffusion_checkpoints/diffusion_cpu/model_000020.pt`,
  reached best validation diffusion loss `6.038887`, and completed a sample
  pass.
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
- The sampler now also supports low-confidence remasking and block-wise
  generation. Low-confidence remasking did not improve the current checkpoints;
  `block_size=4` and `block_size=8` produced more continuous prose than full
  answer-window denoising, but still failed code and factual prompts.
- Base training now exposes the Milestone 3 masking-objective knobs
  `--mask-max-prob` and `--no-mask-loss-reweight`, so the next fresh runs can
  compare the current LLaDA/MDLM-style objective with capped masking or no
  `/ p_mask` weighting.
- Base training now also exposes `--mask-pattern=suffix`, which keeps a random
  prefix visible and trains masked diffusion only on the suffix. This is the
  next train/sample alignment candidate because the public samples are
  fixed-prompt continuations.
- Resuming the 10-shard d20 seq-2048 baseline from step 5000 toward 10k was
  stopped after validation worsened through roughly step 7500. More steps on
  the same objective did not look promising.
- A fresh 8xA100 no-loss-reweight candidate
  `diffusion_a100_d20_s2048_5k_noreweight` was stopped at step 3000 after the
  validation curve reached its best value at step 2000 and then regressed:
  `5.062482 -> 2.390366 -> 2.281531 -> 2.477968 -> 2.209973 -> 2.386717 ->
  2.340456`. Step-2000 and step-3000 samples remained repetitive, so this is a
  rejected recipe, not the selected baseline.
- A suffix-objective candidate `diffusion_a100_d20_s2048_2k_suffix` was trained
  to step 2000, then resumed to step 5000 on 2026-05-20/21 with
  `MASK_PATTERN=suffix`, `PREFIX_MIN_FRAC=0.25`, and `PREFIX_MAX_FRAC=0.75`.
  The resume run completed in 167.58 minutes on 8xA100, reached a minimum
  validation diffusion loss of `1.687262`, and wrote
  `/data2/nanodiffusion/baseline_a100_10s_d20_5k/report/diffusion_a100_d20_s2048_2k_suffix-20260520-222324.md`.
  Despite the lower loss, step-5000 fixed-prompt samples still had topic loops,
  factual drift, and code-prompt failures, so this is useful objective evidence
  but still not the selected quality baseline.
- A fresh 20-shard suffix candidate
  `diffusion_a100_d20_s2048_5k_suffix_20s` completed on 2026-05-21 with the
  same d20 seq-2048 setup. It reached a slightly lower minimum validation loss
  of `1.668640` after 167.35 minutes and wrote
  `/data2/nanodiffusion/baseline_a100_10s_d20_5k/report/diffusion_a100_d20_s2048_5k_suffix_20s-20260521-001927.md`.
  The samples still failed the quality gate with list-like repetitions,
  malformed factual continuations, and unusable `def fibonacci` output. More
  shards alone did not produce the selected baseline.
- A 20-shard seq-1024 suffix candidate
  `diffusion_a100_d20_s1024_5k_suffix_20s` completed on 2026-05-21. It trained
  faster than seq-2048, reached the best validation loss so far (`1.638211`
  after 157.12 minutes), and wrote
  `/data2/nanodiffusion/baseline_a100_10s_d20_5k/report/diffusion_a100_d20_s1024_5k_suffix_20s-20260521-033427.md`.
  The step-5000 samples still failed the quality gate: ordinary prompts looped
  around prompt words, and the `def fibonacci(n):` prompt repeated malformed
  function names instead of code.
- Resuming the seq-1024 suffix checkpoint toward 10k was stopped at step 7000
  after validation regressed (`1.736814 -> 1.951241 -> 1.934933 -> 1.897121 ->
  1.816883`). A separate step-7000 sample report at
  `/data2/nanodiffusion/baseline_a100_10s_d20_5k/report/diffusion_a100_d20_s1024_5k_suffix_20s-step7000-samples-20260521.md`
  showed the same function-name and phrase-loop failures. More steps on this
  recipe did not justify continuing to 10k.
- A short-prefix seq-1024 suffix variant
  `diffusion_a100_d20_s1024_5k_suffix_20s_p025` used
  `PREFIX_MIN_FRAC=0.0` and `PREFIX_MAX_FRAC=0.25` to better match very short
  fixed prompts. It was stopped at step 1000 because validation remained far
  worse (`9.108521 -> 3.620450 -> 3.457720`) and a step-1000 sample report at
  `/data2/nanodiffusion/baseline_a100_10s_d20_5k/report/diffusion_a100_d20_s1024_5k_suffix_20s_p025-step1000-samples-20260521.md`
  still showed prompt-word loops and malformed function-name continuations.
- A seq-256 suffix candidate `diffusion_a100_d20_s256_5k_suffix_20s` was
  stopped at step 3000. It ran faster, but validation remained worse than the
  seq-1024 candidate (`5.203685 -> 2.278308 -> 2.139555 -> 2.104049 ->
  2.127015 -> 2.033860 -> 2.013864`), and a step-3000 sample report at
  `/data2/nanodiffusion/baseline_a100_10s_d20_5k/report/diffusion_a100_d20_s256_5k_suffix_20s-step3000-samples-20260521.md`
  showed the same prompt-word loops and worse code-prompt degeneration.
- A bounded continuation-span objective was added in commit `5b0a41e` and then
  run as `diffusion_a100_d20_s1024_5k_suffix_span_20s` with
  `MASK_PATTERN=suffix_span` and `SPAN_TOKENS=128`. It completed on
  2026-05-21 after 157.01 minutes, with validation improving through step 5000
  (`1.308891 -> 0.625595 -> 0.575395 -> 0.572878 -> 0.555393 -> 0.531591 ->
  0.507425 -> 0.508425 -> 0.493264 -> 0.489348 -> 0.473883`) and final eval
  loss `0.466233`. The report is
  `/data2/nanodiffusion/baseline_a100_10s_d20_5k/report/diffusion_a100_d20_s1024_5k_suffix_span_20s-20260521-095859.md`.
  The lower loss is not directly comparable with whole-suffix losses because
  only a bounded span contributes targets. Fixed-prompt samples still failed:
  `The capital of France is` looped around "capital", and `def fibonacci(n):`
  remained non-code. This is a rejected recipe, not the selected baseline.
- A high-mask version of the span objective
  `diffusion_a100_d20_s1024_5k_suffix_span_m070_20s` used `MASK_EPS=0.7` to
  better match the all-masked start of sampling. It was stopped at step 1000
  after validation remained worse than the default span run (`1.300002 ->
  0.786628 -> 0.737556`) and step-1000 samples showed more severe fixed-prompt
  degeneration. The sample report is
  `/data2/nanodiffusion/baseline_a100_10s_d20_5k/report/diffusion_a100_d20_s1024_5k_suffix_span_m070_20s-step1000-samples-20260521.md`.
- The next candidate should keep the bounded span objective but normalize loss
  by eligible target tokens instead of the full sequence length. The previous
  span runs used `span_tokens=128` on `seq_len=1024`, so the objective was
  scaled down by roughly 8x relative to a full-sequence objective.
- An eligible-normalized bounded-span candidate
  `diffusion_a100_d20_s1024_5k_suffix_span_elig_20s` was run to step 1000. It
  validated the new scaling path (`10.471129 -> 4.889950 -> 4.247498`, final
  eval `4.195501`) but step-1000 samples were still dominated by loops, so it
  was not resumed to 5k. The report is
  `/data2/nanodiffusion/baseline_a100_10s_d20_5k/report/diffusion_a100_d20_s1024_5k_suffix_span_elig_20s-20260521-133041.md`.
- A `block_size=1` sampler spot check on the best seq-1024 suffix, span, and
  eligible-normalized checkpoints did not clear the fixed-prompt gate. The
  suffix checkpoint became slightly more coherent, but factual and code prompts
  still failed.
- The sampler now has a classifier-free guidance candidate (`--cfg-scale`) that
  compares normal prompt conditioning against an unconditional copy where prompt
  tokens are masked. This mirrors a useful LLaDA sampling knob and can be tested
  on existing checkpoints without more base training.
- CFG spot checks on the seq-1024 suffix checkpoints did not clear the sample
  gate. The 20-shard suffix checkpoint with `block_size=4` and `cfg_scale=0`
  occasionally produced a more coherent France/Paris sample, but code prompts
  still failed; `cfg_scale=1.5` and `3.0` usually made samples less stable.
- A 50-shard seq-1024 suffix candidate
  `diffusion_a100_d20_s1024_5k_suffix_50s` completed on 2026-05-21. It reached
  a slightly better step-5000 validation loss than the 20-shard run
  (`1.632275` vs `1.638211`) after 157.43 minutes, with final eval loss
  `1.804405`, and wrote
  `/data2/nanodiffusion/baseline_a100_10s_d20_5k/report/diffusion_a100_d20_s1024_5k_suffix_50s-20260521-140725.md`.
  Fixed-prompt samples still looped and code prompts remained unusable, so this
  is not the selected quality baseline.
- Resuming the 50-shard seq-1024 suffix checkpoint toward 10k was stopped at
  step 5500 after validation regressed to `1.951706`. More steps on this recipe
  are not justified without another objective or sampler change.
- A block-aligned training target
  `diffusion_a100_d20_s1024_5k_block4_20s` used `SPAN_TOKENS=4`,
  `MASK_EPS=0.999`, `MASK_LOSS_REWEIGHT=0`, and
  `LOSS_NORMALIZATION=eligible` so training matched `block_size=4` sampling
  more directly. It was run to step 1000 (`10.401340 -> 7.085275 ->
  6.544755`, final eval `6.592731`) but samples were worse, so it was not
  continued.
- A capped-mask seq-1024 suffix candidate
  `diffusion_a100_d20_s1024_5k_suffix_maxp070_20s` used `MASK_MAX_PROB=0.7`.
  It completed in 157.59 minutes and reached a much lower in-objective
  validation loss (`5.206519 -> 1.745641 -> 1.612728 -> 1.560330 -> 1.543190
  -> 1.443972 -> 1.436938 -> 1.365471 -> 1.344101 -> 1.260612 ->
  1.224907`) with final eval `1.253875`. The report is
  `/data2/nanodiffusion/baseline_a100_10s_d20_5k/report/diffusion_a100_d20_s1024_5k_suffix_maxp070_20s-20260521-175947.md`.
  The capped objective is easier and the samples still failed with factual
  drift and non-code continuations, so this is not the selected baseline.
- The sampler now also supports a fixed left-to-right reveal schedule via
  `--reveal-strategy=left_to_right`. Spot checks on
  `diffusion_a100_d20_s1024_5k_suffix_20s`,
  `diffusion_a100_d20_s1024_5k_suffix_50s`, and
  `diffusion_a100_d20_s1024_5k_suffix_maxp070_20s` showed slightly smoother
  local prose in some cases, but factual prompts still drifted or repeated and
  `def fibonacci(n):` still failed to produce usable code. Reports:
  `/data2/nanodiffusion/baseline_a100_10s_d20_5k/report/diffusion_a100_d20_s1024_5k_suffix_20s-left-to-right-samples-20260521-205727.md`,
  `/data2/nanodiffusion/baseline_a100_10s_d20_5k/report/diffusion_a100_d20_s1024_5k_suffix_50s-left-to-right-samples-20260521-210014.md`,
  and
  `/data2/nanodiffusion/baseline_a100_10s_d20_5k/report/diffusion_a100_d20_s1024_5k_suffix_maxp070_20s-left-to-right-samples-20260521-210145.md`.
- A fully masked suffix objective `MASK_PATTERN=suffix_all` was added to test a
  broader continuation target: random prefix visible, whole suffix masked, and
  all suffix tokens trained with `LOSS_NORMALIZATION=eligible`. The 20-shard
  seq-1024 pilot `diffusion_a100_d20_s1024_1k_suffix_all_20s` ran to step 1000
  in 32.26 minutes. Validation was much worse than the rejected suffix/span
  candidates (`7.367639`, final eval `7.378942`), and samples collapsed into
  France/French/Paris loops plus character-level `def fibonacci(n):`
  degeneration. Report:
  `/data2/nanodiffusion/baseline_a100_10s_d20_5k/report/diffusion_a100_d20_s1024_1k_suffix_all_20s-20260521-210950.md`.
- A smaller d16 seq-1024 suffix pilot
  `diffusion_a100_d16_s1024_1k_suffix_20s` was run as a faster model-size
  diagnostic. It reached step 1000 in 20.88 minutes with minimum validation
  loss `2.019670` and final eval `1.905122`, but fixed-prompt samples still
  failed: France prompts drifted into comparison loops and code prompts repeated
  malformed function names/fragments. Report:
  `/data2/nanodiffusion/baseline_a100_10s_d20_5k/report/diffusion_a100_d16_s1024_1k_suffix_20s-20260521-215254.md`.
- Training now excludes the reserved `[MASK]` id from the reconstruction
  softmax, matching sampling where `[MASK]` is forbidden as an output. A 1k
  d20 seq-1024 suffix pilot
  `diffusion_a100_d20_s1024_1k_suffix_nomasklogit_20s` tested the correction.
  It reached step 1000 in 32.55 minutes with minimum validation loss
  `1.834602` and final eval `1.995015`, but fixed-prompt samples still failed
  with factual drift/repetition and non-code continuations. Report:
  `/data2/nanodiffusion/baseline_a100_10s_d20_5k/report/diffusion_a100_d20_s1024_1k_suffix_nomasklogit_20s-20260521-222757.md`.
- Antithetic mask-probability sampling was added to spread mask probabilities
  across rows in each batch. A 1k d20 seq-1024 suffix pilot
  `diffusion_a100_d20_s1024_1k_suffix_antithetic_20s` used
  `MASK_SAMPLING=antithetic` on top of mask-logit exclusion. It reached step
  1000 in 32.58 minutes with minimum validation loss `1.810764` and final eval
  `1.840987`, slightly better than the uniform mask-sampling pilot. Fixed-prompt
  samples still failed with prompt-word loops and non-code continuations. Report:
  `/data2/nanodiffusion/baseline_a100_10s_d20_5k/report/diffusion_a100_d20_s1024_1k_suffix_antithetic_20s-20260521-231351.md`.
- A fully masked continuation-span objective `MASK_PATTERN=suffix_span_all` was
  added to make the block-wise training target explicit instead of approximating
  it with `suffix_span` and `MASK_EPS=0.999`. It keeps a random prefix visible,
  masks the whole target span, force-masks future suffix tokens without loss,
  and trains only the target span. The 1k d20 seq-1024 pilot
  `diffusion_a100_d20_s1024_1k_suffix_span_all16_20s` used
  `SPAN_TOKENS=16`, `LOSS_NORMALIZATION=eligible`, and
  `MASK_LOSS_REWEIGHT=0`. It reached step 1000 in 32.44 minutes with validation
  loss `10.400656 -> 7.141670 -> 6.765229` and final eval loss `6.730374`.
  Fixed-prompt samples remained dominated by repeated prompt-adjacent words and
  non-code continuations, so this exact block-aligned objective is also
  rejected. Report:
  `/data2/nanodiffusion/baseline_a100_10s_d20_5k/report/diffusion_a100_d20_s1024_1k_suffix_span_all16_20s-20260522-000814.md`.
- A mixed continuation-span objective `MASK_PATTERN=suffix_span_mixed` was added
  to train half the rows with a fully masked span and half with the ordinary
  no-future-leak span. This is a broader curriculum-style objective intended to
  expose the model to both all-mask generation and partially revealed revision.
  The 1k d20 seq-1024 pilot
  `diffusion_a100_d20_s1024_1k_suffix_span_mixed64_20s` used
  `SPAN_TOKENS=64`, `LOSS_NORMALIZATION=eligible`, and
  `MASK_LOSS_REWEIGHT=0`. It reached step 1000 in 32.41 minutes with validation
  loss `7.670461 -> 5.141562 -> 4.825332` and final eval loss `4.672655`,
  better than the pure fully masked span pilot but still worse than the suffix
  pilots. Fixed-prompt samples still failed with prompt-word loops and non-code
  continuations, so it should not be continued to 5k without another broader
  change. Report:
  `/data2/nanodiffusion/baseline_a100_10s_d20_5k/report/diffusion_a100_d20_s1024_1k_suffix_span_mixed64_20s-20260522-004946.md`.
- A corrected full-objective control
  `diffusion_a100_d20_s1024_1k_full_antithetic_20s` was run after mask-logit
  exclusion and antithetic mask sampling were both available. It used
  `MASK_PATTERN=full`, `MASK_SAMPLING=antithetic`, 20 shards, and seq-1024. It
  reached step 1000 in 32.57 minutes with validation loss
  `10.446225 -> 4.025616 -> 3.551752` and final eval loss `3.602521`.
  Fixed-prompt samples still repeated prompt-adjacent words and the code prompt
  did not produce usable code. This control suggests the current failure is not
  only a suffix-objective artifact. Report:
  `/data2/nanodiffusion/baseline_a100_10s_d20_5k/report/diffusion_a100_d20_s1024_1k_full_antithetic_20s-20260522-013001.md`.
- An autoregressive control `ar_d20_s1024_1k_20s_control` was trained with the
  same d20, seq-1024, 20-shard data path and 524,288-token global batch. It
  reached step 1000 in 17.21 minutes with validation BPB
  `3.171669 -> 0.943486 -> 0.857841` and generated substantially more
  language-like fixed-prompt continuations than the diffusion pilots: the France
  prompt answered Paris before repeating, the story/news prompts produced
  coherent prose, and the science/factual prompts were locally plausible. It
  still repeated heavily and did not solve the `def fibonacci(n):` code prompt,
  but this control shows that the shared data, tokenizer, and model scale can
  support basic language modeling after 1k steps. The main remaining bottleneck
  is therefore the diffusion objective/sampler, not the inherited data path
  alone. Log:
  `/data2/nanodiffusion/baseline_a100_10s_d20_5k/logs/ar_d20_s1024_1k_20s_control-20260522-021154.train.log`.
- The sampler now exposes `--remask-strategy=random` in addition to the existing
  confidence and low-confidence paths. Spot checks on
  `diffusion_a100_d20_s1024_5k_suffix_20s` step 5000 and
  `diffusion_a100_d20_s1024_1k_full_antithetic_20s` step 1000 showed that
  random remasking can sometimes move factual prompts toward Paris-like
  continuations, but it also introduces more noisy fragments and does not fix
  the code prompt. It is useful as a comparison recipe, not a selected default.
  Reports:
  `/data2/nanodiffusion/baseline_a100_10s_d20_5k/report/diffusion_a100_d20_s1024_5k_suffix_20s-random-remask-samples-20260522.md`
  and
  `/data2/nanodiffusion/baseline_a100_10s_d20_5k/report/diffusion_a100_d20_s1024_1k_full_antithetic_20s-random-remask-samples-20260522.md`.
- A SEDD-inspired absorbing score-entropy objective is available via
  `--loss-objective=score_entropy`. It interprets model outputs as log score
  ratios for clean non-mask tokens and keeps the existing cross-entropy
  objective as the default. A 1k d20 seq-1024 full-objective pilot
  `diffusion_a100_d20_s1024_1k_score_entropy_full_20s` used
  `LOSS_OBJECTIVE=score_entropy`, `MASK_MAX_PROB=0.999`, and
  `MASK_SAMPLING=antithetic`. The objective was numerically stable after an
  initially very large scale and reached validation loss
  `164823.796875 -> 6.875257 -> 4.694990`, with final eval loss `4.742321`,
  after 32.78 minutes. Fixed-prompt samples still failed: factual prompts
  drifted into repeated country/topic loops and the code prompt degenerated into
  symbols or fragments. This direct score-entropy integration is useful
  infrastructure but not a selected baseline without a matching sampler or
  parameterization change. Report:
  `/data2/nanodiffusion/baseline_a100_10s_d20_5k/report/diffusion_a100_d20_s1024_1k_score_entropy_full_20s-20260522-024848.md`.
- The score-entropy path now has an explicit
  `--score-parameterization=sigma_scaled` option for a closer SEDD-style loss
  parameterization. It subtracts the absorbing-noise scale
  `log(expm1(sigma))` and the non-mask vocabulary prior from score logits
  before the score-entropy terms, while leaving `raw` as the default. A 1k d20
  seq-1024 pilot
  `diffusion_a100_d20_s1024_1k_score_entropy_scaled_full_20s` used
  `LOSS_OBJECTIVE=score_entropy`, `SCORE_PARAMETERIZATION=sigma_scaled`,
  `MASK_MAX_PROB=0.999`, and `MASK_SAMPLING=antithetic`. The scaling fixed the
  raw objective's huge initial loss scale and reached validation loss
  `10.423655 -> 4.129064 -> 3.590117`, with final eval loss `3.636929`, after
  33.09 minutes. Fixed-prompt samples still failed: the France prompt produced
  repeated "capital of ..." variants instead of Paris, and the code prompt
  produced number/topic lists rather than executable Fibonacci code. This is a
  stability improvement, not a selected baseline. Report:
  `/data2/nanodiffusion/baseline_a100_10s_d20_5k/report/diffusion_a100_d20_s1024_1k_score_entropy_scaled_full_20s-20260522-033603.md`.
- `DIFFUSION_SIGMA_CONDITIONING=1` is now available for the next score-entropy
  pilot. It adds a learned scalar `sigma` conditioning projection to the
  bidirectional denoiser and passes the current noise level during both training
  and iterative sampling. This is closer to SEDD's noise-conditioned denoiser
  than loss-only score scaling, while staying disabled by default for old
  checkpoint compatibility. Next 1k candidate:
  `LOSS_OBJECTIVE=score_entropy`, `SCORE_PARAMETERIZATION=sigma_scaled`,
  `DIFFUSION_SIGMA_CONDITIONING=1`, `MASK_MAX_PROB=0.999`,
  `MASK_SAMPLING=antithetic`.
- The 1k d20 seq-1024 pilot
  `diffusion_a100_d20_s1024_1k_score_entropy_sigma_cond_full_20s` used that
  sigma-conditioned recipe. It was stable and reached validation loss
  `10.423655 -> 4.112162 -> 3.580559`, with final eval loss `3.630645`, after
  33.07 minutes. Samples still failed: the France prompt drifted into
  percentage/GDP fragments instead of Paris, and the Fibonacci prompt remained
  non-code. This rules out input-level scalar sigma conditioning as a sufficient
  fix; the next candidate should be deeper denoiser conditioning or a true
  SEDD-style reverse sampler, not another 1k loss-only sweep. Report:
  `/data2/nanodiffusion/baseline_a100_10s_d20_5k/report/diffusion_a100_d20_s1024_1k_score_entropy_sigma_cond_full_20s-20260522-041837.md`.
- `--sampler=sedd_analytic` is available as an opt-in sampling path. It treats
  score-entropy model outputs as score ratios and applies SEDD-style absorbing
  analytic reverse transitions while preserving fixed prompt tokens. The regular
  iterative confidence-reveal sampler remains the default. Next validation should
  be sample-only on existing score-entropy checkpoints before any further
  training run.
- A sample-only report on
  `diffusion_a100_d20_s1024_1k_score_entropy_sigma_cond_full_20s` with the
  `sedd_analytic` recipe did not clear the sample gate. It improved some sentence
  continuity on prose prompts, but the France prompt drifted into unrelated
  road/supply text instead of Paris and the Fibonacci prompt remained non-code.
  Keep this as a diagnostic sampler rather than the selected default. Report:
  `/data2/nanodiffusion/baseline_a100_10s_d20_5k/report/diffusion_a100_d20_s1024_1k_score_entropy_sigma_cond_full_20s-sedd-sampler-20260522-050123.md`.
- The SEDD analytic sampler now uses the same hard categorical exponential-race
  sampling style as the SEDD reference instead of normalized multinomial
  sampling. A sample-only report on the per-layer sigma checkpoint still failed:
  the France prompt drifted into Germany/road/supply text and the Fibonacci
  prompt stayed symbolic/non-code. Report:
  `/data2/nanodiffusion/baseline_a100_10s_d20_5k/report/diffusion_a100_d20_s1024_1k_score_entropy_sigma_layer_full_20s-sedd-hard-sampler-20260522-054824.md`.
- Strict left-to-right single-token reveal was checked sample-only on
  `diffusion_a100_d20_s1024_1k_full_antithetic_20s`,
  `diffusion_a100_d20_s1024_1k_score_entropy_sigma_layer_full_20s`, and
  `diffusion_a100_d20_s1024_5k_suffix_20s`. It made some France continuations
  more grammatical but still looped on "capital of France" and did not produce
  Fibonacci code. This suggests parallel confidence reveal is not the primary
  bottleneck.
- `DIFFUSION_SIGMA_LAYER_CONDITIONING=1` is available as the next denoiser
  conditioning pilot. It injects a separate learned scalar `sigma` projection
  before every transformer block, which is closer to SEDD's per-block
  conditioning than the input-only projection. Use it with
  `LOSS_OBJECTIVE=score_entropy`, `SCORE_PARAMETERIZATION=sigma_scaled`,
  `MASK_MAX_PROB=0.999`, and `MASK_SAMPLING=antithetic` for the next 1k A100
  run.
- The 1k d20 seq-1024 pilot
  `diffusion_a100_d20_s1024_1k_score_entropy_sigma_layer_full_20s` used that
  per-layer conditioning recipe. It was stable but did not improve the gate:
  validation loss `10.423655 -> 4.120845 -> 3.602781`, final eval loss
  `3.653965`, after 33.33 minutes. The final France sample still looped on
  "capital of ..." and the Fibonacci prompt stayed non-code. This rules out
  shallow input or per-block scalar sigma injection as sufficient at this scale.
  Report:
  `/data2/nanodiffusion/baseline_a100_10s_d20_5k/report/diffusion_a100_d20_s1024_1k_score_entropy_sigma_layer_full_20s-20260522-050812.md`.
- `DIFFUSION_SIGMA_ADALN_CONDITIONING=1` is available as the next closer SEDD
  conditioning pilot. It uses sigma-derived shift, scale, and gate terms inside
  each attention/MLP block, zero-initialized so the default block behavior is
  unchanged at initialization. Use it with `LOSS_OBJECTIVE=score_entropy`,
  `SCORE_PARAMETERIZATION=sigma_scaled`, `MASK_MAX_PROB=0.999`, and
  `MASK_SAMPLING=antithetic` for the next 1k A100 run.
- The 1k d20 seq-1024 pilot
  `diffusion_a100_d20_s1024_1k_score_entropy_sigma_adaln_full_20s` used that
  AdaLN conditioning recipe. It was stable and reached validation loss
  `10.423655 -> 4.079122 -> 3.612487`, with final eval loss `3.659920`, after
  35.16 minutes. The fixed-prompt samples still failed the quality gate: the
  France prompt produced "The capital of France is gold" and similar malformed
  variants instead of Paris, and `def fibonacci(n):` generated repeated
  fragments, food/topic lists, or numeric stubs rather than executable code.
  This rules out scalar sigma-driven AdaLN as sufficient at this scale. Report:
  `/data2/nanodiffusion/baseline_a100_10s_d20_5k/report/diffusion_a100_d20_s1024_1k_score_entropy_sigma_adaln_full_20s-20260522-055842.md`.
- `DIFFUSION_SIGMA_EMBEDDING=sinusoidal` is available for the next SEDD-style
  parameterizer pilot. It replaces the scalar `log1p(sigma)` feature with a
  high-dimensional sinusoidal continuous-noise embedding and MLP before the
  existing input, per-layer, or AdaLN sigma projections. After the first failed
  AdaLN pilot below, the MLP output was kept at 256 conditioning dimensions
  instead of expanding to `n_embd` before every per-layer projection, reducing
  the d20 seq-1024 sinusoidal AdaLN parameter count from 1,096,090,866 to
  937,493,746. Use it with `DIFFUSION_SIGMA_ADALN_CONDITIONING=1`,
  `LOSS_OBJECTIVE=score_entropy`, `SCORE_PARAMETERIZATION=sigma_scaled`,
  `MASK_MAX_PROB=0.999`, and `MASK_SAMPLING=antithetic` for the next 1k A100
  run.
- The first sinusoidal AdaLN pilot
  `diffusion_a100_d20_s1024_1k_score_entropy_sigma_adaln_sinusoidal_full_20s`
  was stopped after step 500 because it failed to optimize. It used
  `DIFFUSION_SIGMA_ADALN_CONDITIONING=1` and
  `DIFFUSION_SIGMA_EMBEDDING=sinusoidal`, increased model size to
  1,096,090,866 parameters, and stayed near the uniform-token baseline:
  validation loss `10.380174 -> 10.394501`. This is an optimization failure,
  not a sample-quality run; do not continue that original `n_embd`-wide
  conditioning parameterization without reducing the conditioning parameter
  count or changing its initialization/LR.
  Train log:
  `/data2/nanodiffusion/baseline_a100_10s_d20_5k/logs/diffusion_a100_d20_s1024_1k_score_entropy_sigma_adaln_sinusoidal_full_20s-20260522-064641.train.log`.
- The reduced-width sinusoidal AdaLN pilot
  `diffusion_a100_d20_s1024_1k_score_entropy_sigma_adaln_sinusoidal256_full_20s`
  also failed to optimize and was stopped just after step 500. It used the
  937,493,746-parameter version with 39,976,960 sigma-conditioning parameters,
  but produced the same validation curve `10.429010 -> 10.394501`. This rules
  out the current sinusoidal AdaLN parameterization as a useful next A100
  candidate. The next change should be debugged at small scale first, or should
  use a more conservative injection path such as input-only sinusoidal
  conditioning before spending another 8xA100 run. Train log:
  `/data2/nanodiffusion/baseline_a100_10s_d20_5k/logs/diffusion_a100_d20_s1024_1k_score_entropy_sigma_adaln_sinusoidal256_full_20s-20260522-071025.train.log`.
- Three 1-GPU d8 seq-256 200-step diagnostics were run before spending more
  8xA100 time on sinusoidal conditioning. Scalar input-level sigma conditioning
  was the best small diagnostic (`10.392314 -> 7.015034 -> 6.422985`, final eval
  `6.423954`), while sinusoidal input conditioning was worse
  (`10.401109 -> 7.856457 -> 7.816564`, final eval `7.751679`) and sinusoidal
  AdaLN was also worse (`10.339487 -> 7.586989 -> 7.585907`, final eval
  `7.529066`). This suggests sinusoidal conditioning is trainable at small
  scale but not a better next full-size candidate. Reports:
  `/data2/nanodiffusion/baseline_a100_10s_d20_5k/report/diag_d8_s256_200_score_entropy_sigma_scalar_input-20260522-073212.md`,
  `/data2/nanodiffusion/baseline_a100_10s_d20_5k/report/diag_d8_s256_200_score_entropy_sigma_sinusoidal_input-20260522-073212.md`,
  and
  `/data2/nanodiffusion/baseline_a100_10s_d20_5k/report/diag_d8_s256_200_score_entropy_sigma_sinusoidal_adaln-20260522-073212.md`.
- Diffusion base training can now initialize compatible weights from an
  autoregressive base checkpoint via `--init-from-base-model-tag` and
  `--init-from-base-step`. Matching transformer weights are copied directly;
  token/value embeddings and `lm_head` copy shared tokenizer rows and leave the
  new diffusion `[MASK]` row initialized. This is the next broader candidate
  because the `ar_d20_s1024_1k_20s_control` run proved the same data/model can
  learn basic language modeling, while diffusion-from-scratch still fails the
  fixed-prompt sample gate.
- A 1-GPU AR-initialization smoke diagnostic
  `diag_d20_s1024_50_arinit_score_entropy_sigma_scalar_input` loaded
  `ar_d20_s1024_1k_20s_control` step 1000, copied 147 tensors, and left only the
  new `diffusion_sigma_proj.weight` initialized from scratch. It trained for 50
  steps with `DIFFUSION_SIGMA_CONDITIONING=1`, `LOSS_OBJECTIVE=score_entropy`,
  and `SCORE_PARAMETERIZATION=sigma_scaled`. The score-entropy validation loss
  started high but adapted quickly (`17.282553 -> 6.574758 -> 5.601721`, final
  eval `5.593843`). Samples still looped on "capital"/"fib", so this is only an
  initialization smoke, but it validates the load path and justifies a full 1k
  8xA100 AR-initialized pilot. Report:
  `/data2/nanodiffusion/baseline_a100_10s_d20_5k/report/diag_d20_s1024_50_arinit_score_entropy_sigma_scalar_input-20260522-073851.md`.
- The full 8xA100 AR-initialized pilot
  `diffusion_a100_d20_s1024_1k_arinit_score_entropy_sigma_cond_full_20s`
  loaded the same `ar_d20_s1024_1k_20s_control` step-1000 checkpoint, then ran
  1k diffusion score-entropy steps on 20 shards with scalar sigma input
  conditioning. It optimized slightly better than the comparable scratch
  diffusion runs (`17.314482 -> 3.832415 -> 3.416431`, final eval
  `3.461712`) in 33.06 minutes at about 261k tokens/sec and 37,060 MiB peak GPU
  memory. The fixed-prompt samples still failed the quality gate: "France"
  loops around "actual capital", "meaning of life" repeats phrase templates,
  and `def fibonacci(n):` does not produce usable code. AR initialization helps
  the loss curve but is not sufficient by itself. Report:
  `/data2/nanodiffusion/baseline_a100_10s_d20_5k/report/diffusion_a100_d20_s1024_1k_arinit_score_entropy_sigma_cond_full_20s-20260522-074332.md`.
- The AR-initialized cross-entropy control
  `diffusion_a100_d20_s1024_1k_arinit_ce_sigma_cond_full_20s` used the same
  checkpoint and full random masking but switched back to the CE objective. It
  completed in 32.59 minutes with about 265k tokens/sec and the same 37,060 MiB
  peak memory. The validation curve was `7.703650 -> 3.732457 -> 3.372206`,
  with final eval `3.417173`. This is a small loss improvement over the
  AR-initialized score-entropy run, but it still fails the fixed-prompt gate:
  the France sample repeats "point of view" phrasing, meaning-of-life samples
  loop through definitions, and Fibonacci samples are not code. This rules out
  plain AR-initialized full-mask CE fine-tuning as sufficient. Report:
  `/data2/nanodiffusion/baseline_a100_10s_d20_5k/report/diffusion_a100_d20_s1024_1k_arinit_ce_sigma_cond_full_20s-20260522-082828.md`.
- The AR-initialized continuation-span run
  `diffusion_a100_d20_s1024_1k_arinit_ce_suffix_span_mixed64_20s` combined the
  same base checkpoint with `MASK_PATTERN=suffix_span_mixed`,
  `SPAN_TOKENS=64`, eligible normalization, and no `/p_mask` reweighting. It
  finished in 32.38 minutes at about 267k tokens/sec and 37,061 MiB peak memory,
  with validation `6.805044 -> 4.686185 -> 4.474930` and final eval
  `4.657811`. This continuation-style objective did not preserve useful AR
  behavior: samples regressed into token/word-root loops such as "capital
  capital", "life life", and "accacc" for Fibonacci. It is worse than full-mask
  AR-initialized CE for the fixed-prompt gate. Report:
  `/data2/nanodiffusion/baseline_a100_10s_d20_5k/report/diffusion_a100_d20_s1024_1k_arinit_ce_suffix_span_mixed64_20s-20260522-090734.md`.
- The A100 speedrun script now exposes optimizer learning rates via
  `EMBEDDING_LR`, `UNEMBEDDING_LR`, `MATRIX_LR`, `SCALAR_LR`, and
  `WEIGHT_DECAY`, and records those values in each report. This made it possible
  to test a low-LR, frozen-transformer-matrix AR-initialized pilot without
  hand-editing the command line.
- The frozen-matrix AR-initialized pilot
  `diffusion_a100_d20_s1024_1k_arinit_ce_sigma_cond_freezematrix_lr10x_full_20s`
  used full-mask CE, scalar sigma conditioning, `MATRIX_LR=0`, and 10x lower
  embedding/head/scalar LRs (`0.03`, `0.0008`, `0.05`). It completed in 32.57
  minutes at the same memory scale, with validation
  `7.703650 -> 4.447061 -> 4.312255` and final eval `4.349539`. It did not
  preserve AR continuation behavior: France loops around "rance/French",
  meaning-of-life repeats "life/meaning", and Fibonacci remains bracket or
  "acc" loops. Freezing transformer matrices plus lower LR is therefore not
  sufficient. Report:
  `/data2/nanodiffusion/baseline_a100_10s_d20_5k/report/diffusion_a100_d20_s1024_1k_arinit_ce_sigma_cond_freezematrix_lr10x_full_20s-20260522-094919.md`.
- A stricter AR-bridge objective `MASK_PATTERN=prefix_next` was added. It keeps a
  random prefix visible, masks all future tokens, and trains only the immediately
  next token after the prefix; fixed-prompt sampling can pair it with
  `SAMPLE_REVEAL_STRATEGY=left_to_right` and `SAMPLE_BLOCK_SIZE=1`.
- The AR-initialized prefix-next pilot
  `diffusion_a100_d20_s1024_1k_arinit_ce_prefix_next_ltr_20s` used the same
  `ar_d20_s1024_1k_20s_control` checkpoint, eligible normalization, no `/p_mask`
  reweighting, antithetic mask sampling, and left-to-right block-1 final
  sampling. It completed in 32.40 minutes at the same memory scale, with
  validation `6.966356 -> 6.108967 -> 5.231133` and final eval `5.526335`.
  This is worse than full-mask AR-initialized CE, and samples still fail the
  gate: the France final sample loops around "system/unit", meaning-of-life
  samples repeat "life", "track", or research boilerplate, and Fibonacci remains
  non-code. Plain next-token CE bridging plus left-to-right reveal is therefore
  not sufficient. Report:
  `/data2/nanodiffusion/baseline_a100_10s_d20_5k/report/diffusion_a100_d20_s1024_1k_arinit_ce_prefix_next_ltr_20s-20260522-103354.md`.
- Prefix-next training now also supports an optional AR teacher KL term:
  `--ar-teacher-model-tag`, `--ar-teacher-step`, `--ar-teacher-kl-weight`, and
  `--ar-teacher-temperature`. The teacher is a causal base checkpoint; for each
  prefix-next target, the diffusion student is trained on both token CE and
  `KL(teacher_next_token_distribution || student_distribution)`.
- The AR-teacher prefix-next pilot
  `diffusion_a100_d20_s1024_1k_arinit_teacherkl1_prefix_next_ltr_20s` used the
  same AR checkpoint for initialization and teacher logits, `AR_TEACHER_KL_WEIGHT=1.0`,
  eligible normalization, no `/p_mask` reweighting, antithetic masks, and
  left-to-right block-1 final sampling. It completed in 42.85 minutes at 43,903
  MiB peak memory, with validation `6.999411 -> 4.873528 -> 4.104960` and final
  eval `4.430857`. This improves over one-hot prefix-next eval loss but still
  fails the sample gate: greedy samples collapse mostly to comma repetition,
  France and meaning-of-life left-to-right samples are weak sentence fragments,
  and Fibonacci remains non-code. A single-token diffusion bridge with direct
  AR next-token KL is therefore not sufficient. Report:
  `/data2/nanodiffusion/baseline_a100_10s_d20_5k/report/diffusion_a100_d20_s1024_1k_arinit_teacherkl1_prefix_next_ltr_20s-20260522-183017.md`.
- The AR teacher KL path was extended from single-token `prefix_next` to fully
  masked continuation spans (`suffix_all` and `suffix_span_all`) so the diffusion
  student can receive AR next-token distribution targets over every token in a
  masked block.
- The AR-teacher 64-token span pilot
  `diffusion_a100_d20_s1024_1k_arinit_teacherkl1_span64_ltr_20s` used the same
  AR checkpoint for initialization and teacher logits, `MASK_PATTERN=suffix_span_all`,
  `SPAN_TOKENS=64`, `AR_TEACHER_KL_WEIGHT=1.0`, eligible normalization, no
  `/p_mask` reweighting, antithetic masks, and left-to-right block-4 final
  sampling. It completed in 43.09 minutes at 44,030 MiB peak memory, with
  validation `9.268512 -> 6.802525 -> 6.603037` and final eval `6.591874`.
  This is worse than single-token teacher KL and fails the sample gate: France
  repeats "capital", meaning-of-life repeats "life/meaning", and Fibonacci
  repeats "acc/on/fib" fragments. Teacher-forced span KL is therefore not a
  viable bridge by itself. Report:
  `/data2/nanodiffusion/baseline_a100_10s_d20_5k/report/diffusion_a100_d20_s1024_1k_arinit_teacherkl1_span64_ltr_20s-20260525-111351.md`.
- Base diffusion training now accepts `--attention-mode={bidirectional,causal}`;
  `ATTENTION_MODE=causal` is exposed in the A100 speedrun and recorded in
  reports. This tests whether a denoiser with the same causal attention structure
  as AR decoding preserves continuation behavior better under prefix-next
  training.
- The causal AR-initialized prefix-next pilot
  `diffusion_a100_d20_s1024_1k_arinit_causal_prefix_next_ltr_20s` used
  `ATTENTION_MODE=causal`, `MASK_PATTERN=prefix_next`, eligible normalization,
  no `/p_mask` reweighting, antithetic masks, scalar sigma conditioning, and
  left-to-right block-1 final sampling. It completed in 31.52 minutes at 37,061
  MiB peak memory, with validation `5.984733 -> 6.103633 -> 5.161875` and final
  eval `5.373872`. Causal attention made training faster and reduced memory, but
  it did not clear the sample gate: France drifts into "largest/China"
  repetition, meaning-of-life loops on "world/life", and Fibonacci remains
  non-code. Causal attention alone is therefore not sufficient. Report:
  `/data2/nanodiffusion/baseline_a100_10s_d20_5k/report/diffusion_a100_d20_s1024_1k_arinit_causal_prefix_next_ltr_20s-20260525-120420.md`.

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

Use the completed 10-shard A100 runs as engineering evidence only. The selected
quality baseline is still open, and SFT should remain blocked until fixed-prompt
base samples are no longer dominated by repeated phrases.

Concrete next run: stop spending A100 time on the same 10-shard data/seq-2048
setup. The suffix objective and 20-shard data run improved validation loss but
did not clear the sample gate, so the next useful Milestone 3 candidate should
change the objective rather than only adding data, steps, or shorter sequences.
The suffix/span objective variants, fully masked suffix training, capped
masking, block-aligned training, CFG sampling, fixed reveal scheduling,
mask-logit exclusion, antithetic mask sampling, exact fully masked continuation
span training, mixed continuation-span training, corrected full-objective
training, random remasking, direct score-entropy training, a d16 model-size
pilot, 50-shard data expansion, SEDD analytic sampling, scalar sigma
conditioning through input, per-layer residual injection, AdaLN, sinusoidal
sigma conditioning, AR-initialized score-entropy fine-tuning,
AR-initialized full-mask CE fine-tuning, and AR-initialized mixed
continuation-span fine-tuning, and frozen-matrix/low-LR AR-initialized
fine-tuning, AR-initialized prefix-next left-to-right bridging, prefix-next
AR-teacher KL, teacher-forced multi-token span KL, and causal prefix-next
diffusion have not cleared the sample gate. More of the same recipe should be
avoided; the next candidate needs a broader change than another
scalar/sinusoidal sweep, another plain AR-initialized full-mask/continuation
run, a simple LR/freeze schedule, single-token next-token CE, or single-token
next-token KL, teacher-forced span KL, or a causal-attention-only swap. The AR
control makes this more specific: the same data/model can learn coherent causal
language modeling, but the current diffusion objective and sampler still destroy
continuation quality. The next useful work should change the bridge between AR
and diffusion more fundamentally, for example by training against sampled AR
rollouts instead of teacher-forced gold spans, or by keeping more of the AR
decode trajectory intact during diffusion training.
