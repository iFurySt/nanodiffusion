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


def make_masked_batch(
    clean_ids,
    mask_token_id,
    eps=1e-3,
    generator=None,
    eligible_mask=None,
    max_mask_prob=1.0,
    force_mask=None,
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

    Returns:
        DiffusionBatch where targets are -1 for unmasked positions.
    """
    assert clean_ids.dtype == torch.long
    assert clean_ids.ndim == 2
    assert eps > 0 and eps < 1
    assert max_mask_prob > eps and max_mask_prob <= 1
    B, T = clean_ids.shape
    device = clean_ids.device

    t = torch.rand((B, 1), device=device, generator=generator)
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
):
    """
    Compute the continuous-time masked diffusion objective.

    The per-token CE is divided by the row's mask probability by default,
    matching the simple LLaDA/MDLM estimator. `max_mask_prob` and
    `loss_reweight` and `loss_normalization` are explicit sweep knobs for the
    first training recipe search.
    """
    assert loss_normalization in {"all", "eligible"}
    force_mask = None
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
    elif mask_pattern == "suffix_span":
        assert eligible_mask is None, "suffix_span mask pattern cannot be combined with explicit eligible_mask"
        effective_eligible_mask, force_mask = make_suffix_span_masks(
            clean_ids,
            span_tokens=span_tokens,
            min_prefix_frac=min_prefix_frac,
            max_prefix_frac=max_prefix_frac,
            generator=generator,
        )
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
    )
    logits = model(batch.input_ids)
    vocab_size = logits.size(-1)
    per_token = F.cross_entropy(
        logits.view(-1, vocab_size),
        batch.targets.view(-1),
        ignore_index=-1,
        reduction="none",
    ).view_as(clean_ids)
    weighted = per_token / batch.mask_prob if loss_reweight else per_token
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

        logits = model(ids)
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
                target_positions = editable[row] if remask_low_confidence else remaining[row]
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
        if remask_low_confidence:
            flat_ids[editable.view(-1)] = flat_sampled[editable.view(-1)]
            keep_count = max(1, -(-(total_editable * (step + 1)) // steps))
            keep_scores = conf.masked_fill(~editable, -1.0).view(-1)
            keep_idx = torch.topk(keep_scores, min(keep_count, total_editable)).indices
            low_confidence = editable.clone().view(-1)
            low_confidence[keep_idx] = False
            flat_ids[low_confidence] = mask_token_id
        else:
            remaining_count = int(remaining.sum().item())
            steps_left = steps - step
            reveal_count = max(1, -(-remaining_count // steps_left))
            reveal_scores = conf.masked_fill(~remaining, -1.0).view(-1)
            reveal_idx = torch.topk(reveal_scores, reveal_count).indices
            flat_ids[reveal_idx] = flat_sampled[reveal_idx]

    return ids[0].tolist()
