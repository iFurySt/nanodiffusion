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


def make_masked_batch(clean_ids, mask_token_id, eps=1e-3, generator=None, eligible_mask=None):
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

    Returns:
        DiffusionBatch where targets are -1 for unmasked positions.
    """
    assert clean_ids.dtype == torch.long
    assert clean_ids.ndim == 2
    assert eps > 0 and eps < 1
    B, T = clean_ids.shape
    device = clean_ids.device

    t = torch.rand((B, 1), device=device, generator=generator)
    mask_prob = eps + (1.0 - eps) * t
    if eligible_mask is None:
        eligible_mask = torch.ones((B, T), dtype=torch.bool, device=device)
    else:
        assert eligible_mask.shape == clean_ids.shape
        eligible_mask = eligible_mask.to(device=device, dtype=torch.bool)

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
    input_ids[mask] = mask_token_id
    targets = torch.full_like(clean_ids, -1)
    targets[mask] = clean_ids[mask]
    return DiffusionBatch(input_ids=input_ids, targets=targets, mask=mask, mask_prob=mask_prob)


def masked_diffusion_loss(model, clean_ids, mask_token_id, eps=1e-3, generator=None, eligible_mask=None):
    """
    Compute the continuous-time masked diffusion objective.

    The per-token CE is divided by the row's mask probability, matching the
    simple LLaDA/MDLM estimator. The final loss is normalized by B*T.
    """
    batch = make_masked_batch(clean_ids, mask_token_id, eps=eps, generator=generator, eligible_mask=eligible_mask)
    logits = model(batch.input_ids)
    vocab_size = logits.size(-1)
    per_token = F.cross_entropy(
        logits.view(-1, vocab_size),
        batch.targets.view(-1),
        ignore_index=-1,
        reduction="none",
    ).view_as(clean_ids)
    weighted = per_token / batch.mask_prob
    loss = weighted.sum() / clean_ids.numel()
    metrics = {
        "loss": loss.detach(),
        "mask_fraction": batch.mask.float().mean().detach(),
        "mask_prob": batch.mask_prob.mean().detach(),
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
):
    """
    Fixed-length iterative denoising sampler.

    Prompt tokens, if supplied, are kept fixed at the left side. The remaining
    positions start as [MASK]. Each step predicts all masked positions and keeps
    the highest-confidence subset until no masks remain.
    """
    assert length > 0
    prompt_tokens = prompt_tokens or []
    assert len(prompt_tokens) <= length
    steps = steps or max(1, length - len(prompt_tokens))
    assert steps > 0
    assert no_repeat_ngram_size >= 0
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
                positions = remaining[row].nonzero(as_tuple=False).flatten()
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

        remaining_count = int(remaining.sum().item())
        steps_left = steps - step
        reveal_count = max(1, -(-remaining_count // steps_left))
        reveal_scores = conf.masked_fill(~remaining, -1.0).view(-1)
        reveal_idx = torch.topk(reveal_scores, reveal_count).indices
        flat_ids = ids.view(-1)
        flat_sampled = sampled.view(-1)
        flat_ids[reveal_idx] = flat_sampled[reveal_idx]

    return ids[0].tolist()
