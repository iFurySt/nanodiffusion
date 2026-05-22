"""
Masked discrete diffusion helpers for nanodiffusion.

The tokenizer stays unchanged. The model gets one extra vocabulary row used only
as the [MASK] id:

    mask_token_id = tokenizer.get_vocab_size()
    model_vocab_size = tokenizer.get_vocab_size() + 1

Training corrupts clean token sequences by replacing a random fraction with
[MASK], then predicts the original token only at masked positions.
"""

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


@dataclass
class DiffusionBatch:
    input_ids: torch.Tensor
    targets: torch.Tensor
    mask: torch.Tensor
    mask_prob: torch.Tensor


def get_mask_token_id(tokenizer):
    return tokenizer.get_vocab_size()


def get_diffusion_vocab_size(tokenizer):
    return tokenizer.get_vocab_size() + 1


def make_suffix_eligible_mask(clean_ids, min_prefix_frac=0.25, max_prefix_frac=0.75, generator=None):
    """
    Select a random visible prefix per row and train only the suffix.

    This is a continuation-style objective: prompt-like prefix tokens stay
    fixed, while the suffix is eligible for masked diffusion loss.
    """
    assert clean_ids.ndim == 2
    assert 0 <= min_prefix_frac <= max_prefix_frac < 1
    B, T = clean_ids.shape
    device = clean_ids.device
    min_prefix = min(T - 1, max(0, round(T * min_prefix_frac)))
    max_prefix = min(T - 1, max(min_prefix, round(T * max_prefix_frac)))
    prefix_lens = torch.randint(min_prefix, max_prefix + 1, (B, 1), device=device, generator=generator)
    positions = torch.arange(T, device=device).view(1, T)
    return positions >= prefix_lens


def make_suffix_span_masks(
    clean_ids,
    span_tokens=128,
    min_prefix_frac=0.25,
    max_prefix_frac=0.75,
    generator=None,
):
    """
    Select a random visible prefix and train only a bounded continuation span.

    Tokens after the span are forced to [MASK] in the model input but are not
    targets, so the bidirectional model cannot condition on clean future suffix
    tokens when learning prompt continuation.
    """
    assert clean_ids.ndim == 2
    assert span_tokens > 0
    assert 0 <= min_prefix_frac <= max_prefix_frac < 1
    B, T = clean_ids.shape
    device = clean_ids.device
    min_prefix = min(T - 1, max(0, round(T * min_prefix_frac)))
    max_prefix = min(T - 1, max(min_prefix, round(T * max_prefix_frac)))
    prefix_lens = torch.randint(min_prefix, max_prefix + 1, (B, 1), device=device, generator=generator)
    positions = torch.arange(T, device=device).view(1, T)
    span_ends = torch.clamp(prefix_lens + span_tokens, max=T)
    eligible_mask = (positions >= prefix_lens) & (positions < span_ends)
    force_mask = positions >= span_ends
    return eligible_mask, force_mask


def make_prefix_next_token_masks(
    clean_ids,
    min_prefix_frac=0.25,
    max_prefix_frac=0.75,
    generator=None,
):
    """
    Select a random visible prefix and train exactly the next token.

    Future tokens are forced to [MASK] but are not targets. This matches a
    strict left-to-right reveal schedule more closely than span objectives.
    """
    assert clean_ids.ndim == 2
    assert 0 <= min_prefix_frac <= max_prefix_frac < 1
    B, T = clean_ids.shape
    device = clean_ids.device
    min_prefix = min(T - 1, max(0, round(T * min_prefix_frac)))
    max_prefix = min(T - 1, max(min_prefix, round(T * max_prefix_frac)))
    prefix_lens = torch.randint(min_prefix, max_prefix + 1, (B, 1), device=device, generator=generator)
    positions = torch.arange(T, device=device).view(1, T)
    eligible_mask = positions == prefix_lens
    force_mask = positions > prefix_lens
    return eligible_mask, force_mask


def make_masked_batch(
    clean_ids,
    mask_token_id,
    eps=1e-3,
    generator=None,
    eligible_mask=None,
    max_mask_prob=1.0,
    force_mask=None,
    mask_all_eligible=False,
    mask_sampling="uniform",
):
    """
    Build one LLaDA/MDLM-style masked batch.

    Args:
        clean_ids: LongTensor of shape (B, T), without mask ids.
        mask_token_id: the reserved mask token id, normally tokenizer vocab size.
        eps: lower bound for mask probability, avoiding division by zero.
        generator: optional torch.Generator for deterministic tests/sampling.
        eligible_mask: optional bool tensor of shape (B, T). Only eligible
            positions can be masked and trained. This is used by SFT to keep
            prompt tokens fixed and train only answer tokens.
        max_mask_prob: upper bound for the sampled mask probability. Keeping
            this below 1.0 is a simple sweep knob for avoiding nearly blank
            inputs.
        force_mask: optional bool tensor of shape (B, T). These positions are
            replaced with [MASK] in the input but are not training targets.
        mask_all_eligible: replace every eligible position with [MASK] and
            train all eligible targets. This is a continuation objective knob
            for avoiding clean suffix leakage. It can also be a bool tensor of
            shape (B, 1) to mix fully masked and randomly masked rows.
        mask_sampling: "uniform" samples each row independently; "antithetic"
            spreads row mask probabilities evenly across the batch.

    Returns:
        DiffusionBatch where targets are -1 for unmasked positions.
    """
    assert clean_ids.dtype == torch.long
    assert clean_ids.ndim == 2
    assert eps > 0 and eps < 1
    assert max_mask_prob > eps and max_mask_prob <= 1
    assert mask_sampling in {"uniform", "antithetic"}
    B, T = clean_ids.shape
    device = clean_ids.device

    if mask_sampling == "uniform":
        t = torch.rand((B, 1), device=device, generator=generator)
    else:
        offset = torch.rand((1, 1), device=device, generator=generator)
        row_offsets = torch.arange(B, device=device, dtype=torch.float32).view(B, 1) / B
        t = (offset + row_offsets) % 1.0
    mask_prob = eps + (max_mask_prob - eps) * t
    if eligible_mask is None:
        eligible_mask = torch.ones((B, T), dtype=torch.bool, device=device)
    else:
        assert eligible_mask.shape == clean_ids.shape
        eligible_mask = eligible_mask.to(device=device, dtype=torch.bool)
    if force_mask is None:
        force_mask = torch.zeros((B, T), dtype=torch.bool, device=device)
    else:
        assert force_mask.shape == clean_ids.shape
        force_mask = force_mask.to(device=device, dtype=torch.bool)

    if isinstance(mask_all_eligible, torch.Tensor):
        row_mask_all = mask_all_eligible.to(device=device, dtype=torch.bool)
        assert row_mask_all.shape == (B, 1)
        random_mask = (torch.rand((B, T), device=device, generator=generator) < mask_prob) & eligible_mask
        mask = torch.where(row_mask_all, eligible_mask, random_mask)
        mask_prob = torch.where(row_mask_all, torch.ones_like(mask_prob), mask_prob)
    elif mask_all_eligible:
        mask = eligible_mask.clone()
        mask_prob = torch.ones((B, 1), device=device)
    else:
        mask = (torch.rand((B, T), device=device, generator=generator) < mask_prob) & eligible_mask

    # Keep every row trainable even for tiny smoke-test batches.
    eligible_rows = eligible_mask.any(dim=1)
    empty_rows = (~mask.any(dim=1)) & eligible_rows
    if empty_rows.any():
        row_ids = empty_rows.nonzero(as_tuple=False).flatten()
        for row_id in row_ids.tolist():
            eligible_positions = eligible_mask[row_id].nonzero(as_tuple=False).flatten()
            pick = torch.randint(len(eligible_positions), (1,), device=device, generator=generator)
            mask[row_id, eligible_positions[pick]] = True

    input_ids = clean_ids.clone()
    input_ids[mask | force_mask] = mask_token_id
    targets = torch.full_like(clean_ids, -1)
    targets[mask] = clean_ids[mask]
    return DiffusionBatch(input_ids=input_ids, targets=targets, mask=mask, mask_prob=mask_prob)


def masked_diffusion_loss(
    model,
    clean_ids,
    mask_token_id,
    eps=1e-3,
    generator=None,
    eligible_mask=None,
    max_mask_prob=1.0,
    loss_reweight=True,
    mask_pattern="full",
    min_prefix_frac=0.25,
    max_prefix_frac=0.75,
    span_tokens=128,
    loss_normalization="all",
    mask_sampling="uniform",
    loss_objective="cross_entropy",
    score_parameterization="raw",
):
    """
    Compute the continuous-time masked diffusion objective.

    The per-token CE is divided by the row's mask probability by default,
    matching the simple LLaDA/MDLM estimator. `max_mask_prob` and
    `loss_reweight` and `loss_normalization` are explicit sweep knobs for the
    first training recipe search.
    """
    assert loss_normalization in {"all", "eligible"}
    assert mask_sampling in {"uniform", "antithetic"}
    assert loss_objective in {"cross_entropy", "score_entropy"}
    assert score_parameterization in {"raw", "sigma_scaled"}
    force_mask = None
    mask_all_eligible = False
    if mask_pattern == "full":
        effective_eligible_mask = eligible_mask
    elif mask_pattern == "suffix":
        assert eligible_mask is None, "suffix mask pattern cannot be combined with explicit eligible_mask"
        effective_eligible_mask = make_suffix_eligible_mask(
            clean_ids,
            min_prefix_frac=min_prefix_frac,
            max_prefix_frac=max_prefix_frac,
            generator=generator,
        )
    elif mask_pattern == "suffix_all":
        assert eligible_mask is None, "suffix_all mask pattern cannot be combined with explicit eligible_mask"
        effective_eligible_mask = make_suffix_eligible_mask(
            clean_ids,
            min_prefix_frac=min_prefix_frac,
            max_prefix_frac=max_prefix_frac,
            generator=generator,
        )
        mask_all_eligible = True
    elif mask_pattern == "suffix_span":
        assert eligible_mask is None, "suffix_span mask pattern cannot be combined with explicit eligible_mask"
        effective_eligible_mask, force_mask = make_suffix_span_masks(
            clean_ids,
            span_tokens=span_tokens,
            min_prefix_frac=min_prefix_frac,
            max_prefix_frac=max_prefix_frac,
            generator=generator,
        )
    elif mask_pattern == "suffix_span_all":
        assert eligible_mask is None, "suffix_span_all mask pattern cannot be combined with explicit eligible_mask"
        effective_eligible_mask, force_mask = make_suffix_span_masks(
            clean_ids,
            span_tokens=span_tokens,
            min_prefix_frac=min_prefix_frac,
            max_prefix_frac=max_prefix_frac,
            generator=generator,
        )
        mask_all_eligible = True
    elif mask_pattern == "suffix_span_mixed":
        assert eligible_mask is None, "suffix_span_mixed mask pattern cannot be combined with explicit eligible_mask"
        effective_eligible_mask, force_mask = make_suffix_span_masks(
            clean_ids,
            span_tokens=span_tokens,
            min_prefix_frac=min_prefix_frac,
            max_prefix_frac=max_prefix_frac,
            generator=generator,
        )
        B = clean_ids.size(0)
        mask_all_eligible = torch.rand((B, 1), device=clean_ids.device, generator=generator) < 0.5
    elif mask_pattern == "prefix_next":
        assert eligible_mask is None, "prefix_next mask pattern cannot be combined with explicit eligible_mask"
        effective_eligible_mask, force_mask = make_prefix_next_token_masks(
            clean_ids,
            min_prefix_frac=min_prefix_frac,
            max_prefix_frac=max_prefix_frac,
            generator=generator,
        )
        mask_all_eligible = True
    else:
        raise ValueError(f"unknown mask_pattern: {mask_pattern}")

    batch = make_masked_batch(
        clean_ids,
        mask_token_id,
        eps=eps,
        generator=generator,
        eligible_mask=effective_eligible_mask,
        max_mask_prob=max_mask_prob,
        force_mask=force_mask,
        mask_all_eligible=mask_all_eligible,
        mask_sampling=mask_sampling,
    )
    model_sigma = None
    if (
        getattr(model.config, "diffusion_sigma_conditioning", False)
        or getattr(model.config, "diffusion_sigma_layer_conditioning", False)
        or getattr(model.config, "diffusion_sigma_adaln_conditioning", False)
    ):
        model_sigma = -torch.log1p(-batch.mask_prob.clamp(max=1 - 1e-5))
    logits = model(batch.input_ids, diffusion_sigma=model_sigma) if model_sigma is not None else model(batch.input_ids)
    if loss_objective == "cross_entropy":
        logits[..., mask_token_id] = -float("inf")
        vocab_size = logits.size(-1)
        per_token = F.cross_entropy(
            logits.view(-1, vocab_size),
            batch.targets.view(-1),
            ignore_index=-1,
            reduction="none",
        ).view_as(clean_ids)
        weighted = per_token / batch.mask_prob if loss_reweight else per_token
    else:
        if mask_all_eligible is True or isinstance(mask_all_eligible, torch.Tensor):
            raise ValueError("score_entropy objective does not support fully masked eligible rows")
        mask_prob = batch.mask_prob.clamp(max=1 - 1e-5)
        sigma = model_sigma if model_sigma is not None else -torch.log1p(-mask_prob)
        esigm1 = torch.expm1(sigma).clamp_min(1e-8)
        if score_parameterization == "sigma_scaled":
            esigm1_log = torch.where(
                sigma < 0.5,
                torch.expm1(sigma),
                sigma.exp() - 1,
            ).clamp_min(1e-8).log()
            logits = logits - esigm1_log[..., None] - math.log(mask_token_id)
        ratio = 1 / esigm1
        # The model output is interpreted as log score ratios. For absorbing
        # diffusion, only non-mask vocabulary states contribute to the positive
        # term; the absorbing [MASK] state is the current corrupted state.
        pos_term = logits[..., :mask_token_id].exp().sum(dim=-1)
        safe_targets = batch.targets.clamp_min(0)
        clean_log_score = logits.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
        per_token = pos_term - ratio * clean_log_score + ratio * (ratio.log() - 1)
        per_token = per_token.masked_fill(~batch.mask, 0.0)
        dsigma = (max_mask_prob - eps) / (1 - mask_prob)
        weighted = dsigma * per_token
    if loss_normalization == "eligible":
        if effective_eligible_mask is None:
            denominator = torch.tensor(clean_ids.numel(), device=clean_ids.device, dtype=weighted.dtype)
            eligible_fraction = torch.tensor(1.0, device=clean_ids.device, dtype=weighted.dtype)
        else:
            denominator = effective_eligible_mask.sum().clamp_min(1).to(dtype=weighted.dtype)
            eligible_fraction = effective_eligible_mask.float().mean()
    else:
        denominator = torch.tensor(clean_ids.numel(), device=clean_ids.device, dtype=weighted.dtype)
        if effective_eligible_mask is None:
            eligible_fraction = torch.tensor(1.0, device=clean_ids.device, dtype=weighted.dtype)
        else:
            eligible_fraction = effective_eligible_mask.float().mean()
    loss = weighted.sum() / denominator
    metrics = {
        "loss": loss.detach(),
        "mask_fraction": batch.mask.float().mean().detach(),
        "mask_prob": batch.mask_prob.mean().detach(),
        "eligible_fraction": eligible_fraction.detach(),
    }
    return loss, metrics


def _banned_ngram_tokens(row_ids, mask_token_id, ngram_size, position):
    if ngram_size <= 0:
        return set()
    if ngram_size == 1:
        return {int(tok) for tok in row_ids.tolist() if int(tok) != mask_token_id}
    if position < ngram_size - 1:
        return set()

    prefix = row_ids[position - ngram_size + 1 : position]
    if (prefix == mask_token_id).any():
        return set()
    prefix = tuple(int(tok) for tok in prefix.tolist())

    banned = set()
    for start in range(0, position - ngram_size + 1):
        ngram = row_ids[start : start + ngram_size]
        if (ngram == mask_token_id).any():
            continue
        if tuple(int(tok) for tok in ngram[:-1].tolist()) == prefix:
            banned.add(int(ngram[-1]))
    return banned


def _model_logits(model, ids, diffusion_sigma=None):
    if diffusion_sigma is None:
        return model(ids)
    return model(ids, diffusion_sigma=diffusion_sigma)


def _apply_score_parameterization(logits, sigma, mask_token_id, score_parameterization):
    if score_parameterization == "raw":
        return logits
    assert score_parameterization == "sigma_scaled"
    esigm1_log = torch.where(
        sigma < 0.5,
        torch.expm1(sigma),
        sigma.exp() - 1,
    ).clamp_min(1e-8).log()
    return logits - esigm1_log[..., None] - math.log(mask_token_id)


def _sample_categorical_weights(weights, generator):
    noise = 1e-10 - (torch.rand(weights.shape, device=weights.device, generator=generator) + 1e-10).log()
    return (weights / noise).argmax(dim=-1)


@torch.inference_mode()
def _sample_sedd_analytic(
    model,
    mask_token_id,
    length,
    prompt_tokens,
    steps,
    seed,
    forbidden_token_ids,
    score_parameterization,
    mask_eps,
    mask_max_prob,
):
    device = model.get_device()
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    ids = torch.full((1, length), mask_token_id, dtype=torch.long, device=device)
    if prompt_tokens:
        prompt = torch.tensor(prompt_tokens, dtype=torch.long, device=device)
        ids[0, :len(prompt_tokens)] = prompt

    editable = torch.ones((1, length), dtype=torch.bool, device=device)
    if prompt_tokens:
        editable[:, :len(prompt_tokens)] = False

    forbidden_token_ids = set(forbidden_token_ids or [])
    forbidden_token_ids.add(mask_token_id)
    forbidden_token_ids = [tok for tok in forbidden_token_ids if 0 <= tok < model.config.vocab_size]

    mask_probs = torch.linspace(mask_max_prob, mask_eps, steps + 1, device=device).clamp(max=1 - 1e-5)
    sigmas = -torch.log1p(-mask_probs)
    for step in range(steps):
        remaining = (ids == mask_token_id) & editable
        if not remaining.any():
            break

        curr_sigma = sigmas[step].view(1, 1)
        next_sigma = sigmas[step + 1].view(1, 1)
        dsigma = curr_sigma - next_sigma
        diffusion_sigma = curr_sigma if (
            getattr(model.config, "diffusion_sigma_conditioning", False)
            or getattr(model.config, "diffusion_sigma_layer_conditioning", False)
            or getattr(model.config, "diffusion_sigma_adaln_conditioning", False)
        ) else None
        log_score = _model_logits(model, ids, diffusion_sigma=diffusion_sigma)
        log_score = _apply_score_parameterization(log_score, curr_sigma, mask_token_id, score_parameterization)
        if forbidden_token_ids:
            log_score[..., forbidden_token_ids] = -float("inf")
        log_score = log_score.scatter(-1, ids.unsqueeze(-1), torch.zeros_like(log_score[..., :1]))
        score = log_score.exp()

        stag_score = score.clone()
        extra_const = (1 - dsigma.exp()) * stag_score.sum(dim=-1)
        stag_score = stag_score * dsigma.exp().view(1, 1, 1)
        stag_score[..., mask_token_id] += extra_const

        transition = torch.zeros_like(stag_score)
        transition.scatter_(-1, ids.unsqueeze(-1), dsigma.neg().exp().view(1, 1, 1).expand_as(ids.unsqueeze(-1)).to(stag_score.dtype))
        mask_transition = (ids == mask_token_id).to(stag_score.dtype).unsqueeze(-1) * (1 - dsigma.neg().exp()).view(1, 1, 1)
        transition = transition + mask_transition
        probs = (stag_score * transition).clamp_min(0)
        sampled = _sample_categorical_weights(probs, rng)
        ids = torch.where(remaining, sampled, ids)

    remaining = (ids == mask_token_id) & editable
    if remaining.any():
        curr_sigma = sigmas[-1].view(1, 1)
        diffusion_sigma = curr_sigma if (
            getattr(model.config, "diffusion_sigma_conditioning", False)
            or getattr(model.config, "diffusion_sigma_layer_conditioning", False)
            or getattr(model.config, "diffusion_sigma_adaln_conditioning", False)
        ) else None
        log_score = _model_logits(model, ids, diffusion_sigma=diffusion_sigma)
        log_score = _apply_score_parameterization(log_score, curr_sigma, mask_token_id, score_parameterization)
        if forbidden_token_ids:
            log_score[..., forbidden_token_ids] = -float("inf")
        log_score = log_score.scatter(-1, ids.unsqueeze(-1), torch.zeros_like(log_score[..., :1]))
        score = log_score.exp()

        stag_score = score.clone()
        extra_const = (1 - curr_sigma.exp()) * stag_score.sum(dim=-1)
        stag_score = stag_score * curr_sigma.exp().view(1, 1, 1)
        stag_score[..., mask_token_id] += extra_const
        transition = torch.zeros_like(stag_score)
        transition.scatter_(-1, ids.unsqueeze(-1), curr_sigma.neg().exp().view(1, 1, 1).expand_as(ids.unsqueeze(-1)).to(stag_score.dtype))
        mask_transition = (ids == mask_token_id).to(stag_score.dtype).unsqueeze(-1) * (1 - curr_sigma.neg().exp()).view(1, 1, 1)
        probs = (stag_score * (transition + mask_transition))[..., :mask_token_id].clamp_min(0)
        sampled = _sample_categorical_weights(probs, rng)
        ids = torch.where(remaining, sampled, ids)

    return ids[0].tolist()


@torch.inference_mode()
def sample_masked_diffusion(
    model,
    mask_token_id,
    length,
    prompt_tokens=None,
    steps=None,
    temperature=0.0,
    top_k=None,
    seed=42,
    forbidden_token_ids=None,
    repeat_penalty=0.0,
    no_repeat_ngram_size=0,
    block_size=0,
    remask_low_confidence=False,
    remask_strategy="none",
    cfg_scale=0.0,
    reveal_strategy="confidence",
    sampler="iterative",
    score_parameterization="raw",
    mask_eps=1e-3,
    mask_max_prob=0.999,
):
    """
    Fixed-length iterative denoising sampler.

    Prompt tokens, if supplied, are kept fixed at the left side. The remaining
    positions start as [MASK]. By default each step predicts all masked
    positions and keeps the highest-confidence subset until no masks remain.
    With low-confidence remasking enabled, every step may revise generated
    positions and only the current highest-confidence subset stays unmasked.
    """
    assert length > 0
    prompt_tokens = prompt_tokens or []
    assert len(prompt_tokens) <= length
    steps = steps or max(1, length - len(prompt_tokens))
    assert steps > 0
    assert no_repeat_ngram_size >= 0
    assert block_size >= 0
    assert cfg_scale >= 0
    assert remask_strategy in {"none", "low_confidence", "random"}
    assert reveal_strategy in {"confidence", "left_to_right"}
    assert sampler in {"iterative", "sedd_analytic"}
    assert score_parameterization in {"raw", "sigma_scaled"}
    assert 0 < mask_eps < mask_max_prob <= 1
    if remask_low_confidence and remask_strategy == "none":
        remask_strategy = "low_confidence"
    if sampler == "sedd_analytic":
        return _sample_sedd_analytic(
            model,
            mask_token_id=mask_token_id,
            length=length,
            prompt_tokens=prompt_tokens,
            steps=steps,
            seed=seed,
            forbidden_token_ids=forbidden_token_ids,
            score_parameterization=score_parameterization,
            mask_eps=mask_eps,
            mask_max_prob=mask_max_prob,
        )

    gen_tokens = length - len(prompt_tokens)
    if block_size > 0 and gen_tokens > block_size:
        output = list(prompt_tokens)
        remaining_tokens = gen_tokens
        while remaining_tokens > 0:
            current_block = min(block_size, remaining_tokens)
            current_length = len(output) + current_block
            block_steps = max(1, round(steps * (current_block / gen_tokens)))
            output = sample_masked_diffusion(
                model,
                mask_token_id=mask_token_id,
                length=current_length,
                prompt_tokens=output,
                steps=block_steps,
                temperature=temperature,
                top_k=top_k,
                seed=seed + len(output),
                forbidden_token_ids=forbidden_token_ids,
                repeat_penalty=repeat_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                block_size=0,
                remask_low_confidence=remask_low_confidence,
                remask_strategy=remask_strategy,
                cfg_scale=cfg_scale,
                reveal_strategy=reveal_strategy,
                sampler=sampler,
                score_parameterization=score_parameterization,
                mask_eps=mask_eps,
                mask_max_prob=mask_max_prob,
            )
            remaining_tokens -= current_block
        return output

    device = model.get_device()
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    ids = torch.full((1, length), mask_token_id, dtype=torch.long, device=device)
    if prompt_tokens:
        prompt = torch.tensor(prompt_tokens, dtype=torch.long, device=device)
        ids[0, :len(prompt_tokens)] = prompt

    editable = torch.ones((1, length), dtype=torch.bool, device=device)
    if prompt_tokens:
        editable[:, :len(prompt_tokens)] = False

    forbidden_token_ids = set(forbidden_token_ids or [])
    forbidden_token_ids.add(mask_token_id)
    forbidden_token_ids = [tok for tok in forbidden_token_ids if 0 <= tok < model.config.vocab_size]

    total_editable = int(editable.sum().item())
    for step in range(steps):
        remaining = (ids == mask_token_id) & editable
        if not remaining.any():
            break

        diffusion_sigma = None
        if (
            getattr(model.config, "diffusion_sigma_conditioning", False)
            or getattr(model.config, "diffusion_sigma_layer_conditioning", False)
            or getattr(model.config, "diffusion_sigma_adaln_conditioning", False)
        ):
            editable_count = editable.sum(dim=1, keepdim=True).clamp_min(1)
            mask_prob = remaining.sum(dim=1, keepdim=True).float() / editable_count.float()
            diffusion_sigma = -torch.log1p(-mask_prob.clamp(max=0.999))
        logits = _model_logits(model, ids, diffusion_sigma=diffusion_sigma)
        if cfg_scale > 0 and prompt_tokens:
            uncond_ids = ids.clone()
            uncond_ids[:, :len(prompt_tokens)] = mask_token_id
            uncond_logits = _model_logits(model, uncond_ids, diffusion_sigma=diffusion_sigma)
            logits = uncond_logits + (cfg_scale + 1) * (logits - uncond_logits)
        if forbidden_token_ids:
            logits[..., forbidden_token_ids] = -float("inf")
        if repeat_penalty > 0:
            generated = editable & (ids != mask_token_id)
            for row in range(ids.size(0)):
                seen = torch.unique(ids[row, generated[row]])
                if seen.numel() > 0:
                    logits[row, :, seen] -= repeat_penalty
        if no_repeat_ngram_size > 0:
            for row in range(ids.size(0)):
                target_positions = editable[row] if remask_strategy != "none" else remaining[row]
                positions = target_positions.nonzero(as_tuple=False).flatten()
                for pos in positions.tolist():
                    banned = _banned_ngram_tokens(ids[row], mask_token_id, no_repeat_ngram_size, pos)
                    if banned:
                        valid = [tok for tok in banned if 0 <= tok < logits.size(-1)]
                        logits[row, pos, valid] = -float("inf")
        if top_k is not None and top_k > 0:
            k = min(top_k, logits.size(-1))
            vals, idx = torch.topk(logits, k, dim=-1)
            filtered = torch.full_like(logits, -float("inf"))
            logits = filtered.scatter(-1, idx, vals)

        if temperature > 0:
            probs = F.softmax(logits / temperature, dim=-1)
            sampled = torch.multinomial(probs.view(-1, probs.size(-1)), 1, generator=rng).view_as(ids)
            conf = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
        else:
            probs = F.softmax(logits, dim=-1)
            conf, sampled = probs.max(dim=-1)

        flat_ids = ids.view(-1)
        flat_sampled = sampled.view(-1)
        if remask_strategy != "none":
            flat_ids[editable.view(-1)] = flat_sampled[editable.view(-1)]
            keep_count = max(1, -(-(total_editable * (step + 1)) // steps))
            if remask_strategy == "random":
                keep_scores = torch.rand(conf.shape, device=device, generator=rng).masked_fill(~editable, -1.0).view(-1)
            else:
                keep_scores = conf.masked_fill(~editable, -1.0).view(-1)
            keep_idx = torch.topk(keep_scores, min(keep_count, total_editable)).indices
            low_confidence = editable.clone().view(-1)
            low_confidence[keep_idx] = False
            flat_ids[low_confidence] = mask_token_id
        else:
            remaining_count = int(remaining.sum().item())
            steps_left = steps - step
            reveal_count = max(1, -(-remaining_count // steps_left))
            if reveal_strategy == "confidence":
                reveal_scores = conf.masked_fill(~remaining, -1.0).view(-1)
                reveal_idx = torch.topk(reveal_scores, reveal_count).indices
            else:
                reveal_idx = remaining.view(-1).nonzero(as_tuple=False).flatten()[:reveal_count]
            flat_ids[reveal_idx] = flat_sampled[reveal_idx]

    return ids[0].tolist()
