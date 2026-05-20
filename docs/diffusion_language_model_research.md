# Diffusion Language Model Research Notes

Date: 2026-05-19

This note evaluates how to fork nanochat into an educational diffusion language
model project with the same spirit as nanochat: one understandable codebase, one
main training script, same dataset/tokenizer pipeline where possible, and a
realistic path toward a GPT-2-level small diffusion Transformer language model.

## Recommendation

Start with a masked discrete diffusion language model, closest to LLaDA and
MDLM, not a continuous embedding diffusion model.

The minimal nanochat fork should:

1. Reuse the ClimbMix parquet dataset and the existing BPE tokenizer.
2. Add one reserved `[MASK]` token outside the tokenizer's normal vocabulary.
3. Train a bidirectional Transformer to reconstruct randomly masked tokens.
4. Use the LLaDA/MDLM continuous-time masked objective:

   ```text
   t ~ Uniform(0, 1)
   p_mask = eps + (1 - eps) * t
   x_t = mask each token of x_0 independently with probability p_mask
   loss = CE(model(x_t)[masked_positions], x_0[masked_positions]) / p_mask
   loss = sum(loss) / (batch_size * sequence_length)
   ```

5. Generate by starting from all masks, repeatedly predicting all masked
   positions, and revealing a scheduled number of high-confidence tokens each
   step. For instruction tuning, keep prompt tokens fixed and diffuse only the
   answer span.

This gives the smallest conceptual diff from nanochat while matching the most
active and scalable recent line of diffusion LMs.

## Why This Path

The current nanochat stack already has the expensive pieces we want to keep:

- Dataset download/cache: `nanochat.dataset`
- BPE tokenizer and token byte accounting: `nanochat.tokenizer`
- Fast DDP training loop, checkpointing, reports, and eval task wrappers
- A compact Transformer implementation with clear scaling rules

The current model is causal. A masked diffusion LM needs a bidirectional denoiser.
So the central code change is not data handling; it is the model attention mask,
training target, and sampler.

## Surveyed Work

### LLaDA: Large Language Diffusion Models

Paper: https://arxiv.org/abs/2502.09992
Code: https://github.com/ML-GSAI/LLaDA

LLaDA is the strongest evidence that masked diffusion can scale to LLM-like
pretraining and SFT. The paper uses a forward masking process and a reverse
generation process where a Transformer predicts masked tokens. The repo's
guidelines show that pretraining can be expressed with only a small loss change:
sample a random mask probability, mask tokens, predict originals at masked
positions, and divide each token loss by its mask probability.

Important implementation details:

- Use a normal Transformer encoder-style denoiser, not a causal decoder.
- Reserve a mask token id.
- For SFT, do not mask the prompt; train only answer tokens.
- Sampling quality is best when denoising steps are close to generated length,
  but block sampling can trade quality for speed.
- Low-confidence remasking is a practical default: predict all masks, keep the
  highest-confidence predictions, and continue denoising the rest.

Fit for nanochat: very high. It is the simplest teaching implementation.

### MDLM: Simple and Effective Masked Diffusion Language Models

Paper: https://arxiv.org/abs/2406.07524
Code: https://github.com/kuleshov-group/mdlm

MDLM shows that masked discrete diffusion can approach autoregressive perplexity
with a simplified objective and modern training recipes. The released code
contains several parameterizations, but the useful baseline for nanochat is the
SUBS/masked parameterization: corrupt tokens to a mask state, predict the clean
tokens, and use a continuous-time loss scaling.

Useful repo details:

- If the tokenizer has no mask token, MDLM uses `mask_index = vocab_size`.
- The forward process samples `x_t` by replacing tokens with the mask id.
- The reverse sampler starts from all masks and iteratively updates masked
  positions.
- The code includes AR, D3PM, SEDD, and masked variants, so it is a good
  reference for evaluation and sampler variants, but it is heavier than we want
  for a nanochat-style fork.

Fit for nanochat: high as a reference, but the implementation should be much
smaller than MDLM.

### SEDD: Score Entropy Discrete Diffusion

Paper: https://arxiv.org/abs/2310.16834
Code: https://github.com/louaaron/Score-Entropy-Discrete-Diffusion

SEDD is theoretically elegant and strong. It estimates probability ratios in
discrete space with a score-entropy loss and reports large improvements over
older diffusion LM methods. The official code samples a corruption level,
perturbs tokens through a graph transition, predicts log scores, and optimizes a
score entropy objective.

Tradeoff:

- Pros: principled discrete diffusion, strong paper results, good infilling
  story.
- Cons: less intuitive for a first teaching implementation. It requires graph
  transitions, ratio/score parameterization, and a less familiar loss.

Fit for nanochat: good for a second implementation track after the masked
diffusion baseline works.

### BD3-LM: Block Diffusion

Paper: https://arxiv.org/abs/2503.09573
Code: https://github.com/kuleshov-group/bd3lms

BD3-LM combines autoregressive generation over blocks with diffusion inside each
block. It addresses two practical problems of pure masked diffusion: fixed-length
generation and slow full-sequence denoising. The repo includes MDLM, SEDD, and
SSD-LM baselines plus block-autoregressive likelihood parameterization and
samplers with KV caching.

Tradeoff:

- Pros: arbitrary-length generation, more practical inference, strong recent
  direction.
- Cons: more moving parts: block-causal masks, block samplers, KV cache logic,
  and data-driven schedules.

Fit for nanochat: strong v2 direction. It should not be the first milestone if
the goal is educational clarity.

### D3PM

Paper: https://arxiv.org/abs/2107.03006

D3PM is the older discrete diffusion foundation: transition matrices over
discrete states, including absorbing mask states. It is important background, but
large-vocabulary language modeling with full transition matrices is not the
cleanest route for this fork.

Fit for nanochat: background only. Use the absorbing-mask idea, not a full D3PM
implementation.

### Diffusion-LM and SSD-LM

Diffusion-LM paper: https://arxiv.org/abs/2205.14217
Diffusion-LM code: https://github.com/XiangLi1999/Diffusion-LM

SSD-LM paper: https://arxiv.org/abs/2210.17432
SSD-LM code: https://github.com/xhan77/ssd-lm

These methods diffuse in continuous embedding/simplex spaces and are useful for
controllable generation. They are not the best first fit for nanochat because
they add rounding, embedding projection, simplex geometry, or auxiliary control
machinery. That complexity fights the goal of a clean GPT-2-level teaching
implementation.

Fit for nanochat: not recommended for the initial fork.

## Proposed Nanochat Architecture Changes

Keep the existing `GPT` class available for AR nanochat. Add a diffusion variant
instead of overloading every path:

- `nanochat/diffusion_gpt.py`
  - either subclass or lightly fork `GPT`
  - add `attention_mode = "bidirectional"` or directly call attention with
    `causal=False`
  - disable KV-cache generation paths for diffusion
  - use full-context attention by default

- `nanochat/flash_attention.py`
  - support `causal=False` in the SDPA fallback path
  - for bidirectional training, call
    `F.scaled_dot_product_attention(..., is_causal=False)`

- Token vocabulary:
  - `base_vocab_size = tokenizer.get_vocab_size()`
  - `mask_token_id = base_vocab_size`
  - diffusion model `vocab_size = base_vocab_size + 1`
  - ignore or force `-inf` on the mask logit during reconstruction/sampling

The tokenizer does not need to learn the mask token. It only needs a stable id
known to the diffusion scripts and checkpoint metadata.

## Proposed Training Scripts

Add parallel scripts rather than replacing AR scripts:

- `scripts/diffusion_base_train.py`
- `scripts/diffusion_base_eval.py`
- `scripts/diffusion_chat_sft.py`
- `scripts/diffusion_chat_eval.py`
- `scripts/diffusion_chat_cli.py`
- `runs/diffusion_speedrun.sh`

The first milestone should only implement:

- tokenizer reuse
- base masked diffusion pretraining
- unconditional/fixed-length text sampling
- checkpoint save/load
- a simple report section

SFT and chat UI should come after base model samples are intelligible.

## Base Training Loop Sketch

Use the existing pretraining dataloader, but treat `inputs` as the clean sequence
`x0` instead of using shifted next-token targets.

```python
inputs, _, dataloader_state = next(train_loader)
x0 = inputs

t = torch.rand(x0.size(0), device=x0.device)
p_mask = eps + (1 - eps) * t
p = p_mask[:, None]
mask = torch.rand_like(x0.float()) < p

# avoid all-unmasked rows for stable logging
mask[:, 0] = True

xt = torch.where(mask, mask_token_id, x0)
logits = model(xt)
logits[..., mask_token_id] = -torch.inf

token_loss = F.cross_entropy(
    logits[mask],
    x0[mask],
    reduction="none",
) / p_mask[:, None].expand_as(x0)[mask]

loss = token_loss.sum() / x0.numel()
```

Useful defaults:

- `eps = 1e-3`
- sample `t` per row, not per token
- antithetic `t` sampling later if variance is noisy
- keep nanochat's existing Muon/AdamW optimizer initially
- start with `window_pattern="L"` for A100 and for bidirectional attention

## Sampling Sketch

For unconditional samples of length `T`:

```python
x = torch.full((batch, T), mask_token_id, device=device)
for step in range(num_steps):
    logits = model(x)
    logits[..., mask_token_id] = -torch.inf
    x0 = sample_or_argmax(logits, temperature)
    confidence = softmax(logits).gather(-1, x0[..., None]).squeeze(-1)
    confidence[x != mask_token_id] = -inf
    reveal top-k masked positions for this step
    x[reveal] = x0[reveal]
```

For prompt-conditioned generation:

- create `[prompt tokens] + [MASK] * gen_length`
- never change prompt positions
- optionally denoise one block at a time, e.g. 32 or 64 tokens per block
- default to low-confidence remasking

## Evaluation Plan

Autoregressive bpb is not directly comparable to diffusion NELBO, so we should
report two metric families:

1. Training/eval diffusion loss:
   - validation masked NELBO estimate
   - bytes-normalized version using existing `token_bytes`

2. Generation/task metrics:
   - sample quality with a GPT-2 evaluator if available
   - CORE-style tasks via generation
   - multiple-choice tasks via Monte Carlo pseudo-likelihood, following the
     LLaDA `get_log_likelihood.py` pattern

For the educational speedrun, the user-facing target should be "GPT-2-level task
capability" rather than "same bpb number as AR nanochat".

## Milestones

### Milestone 1: Minimal base DLM

- Add mask id and bidirectional model path.
- Train d12 or d16 smoke models.
- Generate fixed-length samples.
- Validate that loss decreases and samples stop being random.

### Milestone 2: GPT-2-level base run

- Scale to d24/d26 on 8xA100/H100.
- Tune total batch size, mask schedule, denoising steps, and sampling strategy.
- Add diffusion-specific report metrics.

### Milestone 3: SFT chat model

- Port `scripts/chat_sft.py` data rendering.
- Keep prompts unmasked; mask and train only assistant spans.
- Implement `diffusion_chat_cli.py` and then web UI.

### Milestone 4: Advanced samplers

- Low-confidence sampling.
- Block sampling for flexible length.
- Optional classifier-free guidance for instruction responses.
- Optional SEDD objective experiment.

## Open Risks

- Bidirectional full-context attention is more expensive than causal attention
  with KV cache. Training should be fine, but inference can need many model
  calls per generated block.
- The current nanochat architecture has directional details such as smear and
  causal KV-cache generation. They may be acceptable initially, but the clean
  diffusion fork should make the bidirectional behavior explicit.
- Evaluation can become confusing because diffusion likelihood estimates and AR
  bpb are not identical. The report must label metrics clearly.
- Pure masked diffusion tends to prefer fixed generation length. Block diffusion
  is probably needed for a polished chat UX.

## Final Decision

Implement an LLaDA/MDLM-style masked diffusion Transformer first.

Do not start with SEDD, BD3-LM, Diffusion-LM, or SSD-LM. They are valuable
references, but they add complexity before the basic teaching story is clear.

The first fork should be "nanochat, but `base_train` becomes denoising masked
tokens with a bidirectional Transformer." Once that works end to end, add SFT
and block sampling.

The current public recipe lives in:

- `runs/diffusion_speedrun_a100.sh` for base masked diffusion pretraining.
- `scripts/diffusion_sample_sweep.py` for fixed-prompt sampling comparisons.
- `runs/diffusion_sft_a100.sh` for prompt-fixed, answer-only diffusion SFT.

What is borrowed from LLaDA/MDLM: reserved mask id, randomly sampled mask
probabilities, bidirectional denoising, response-only SFT, and confidence-based
iterative generation. What is simplified for NanoDiffusion: fixed-length
generation, a compact nanochat-style Transformer, simple top-k/temperature
sampling, and a small repeat penalty instead of a full remasking scheduler.

## Source Index

- LLaDA paper: https://arxiv.org/abs/2502.09992
- LLaDA code: https://github.com/ML-GSAI/LLaDA
- MDLM paper: https://arxiv.org/abs/2406.07524
- MDLM code: https://github.com/kuleshov-group/mdlm
- SEDD paper: https://arxiv.org/abs/2310.16834
- SEDD code: https://github.com/louaaron/Score-Entropy-Discrete-Diffusion
- BD3-LM paper: https://arxiv.org/abs/2503.09573
- BD3-LM code: https://github.com/kuleshov-group/bd3lms
- D3PM paper: https://arxiv.org/abs/2107.03006
- Diffusion-LM paper: https://arxiv.org/abs/2205.14217
- Diffusion-LM code: https://github.com/XiangLi1999/Diffusion-LM
- SSD-LM paper: https://arxiv.org/abs/2210.17432
- SSD-LM code: https://github.com/xhan77/ssd-lm
