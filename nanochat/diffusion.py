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


def make_masked_batch(clean_ids, mask_token_id, eps=1e-3, generator=None):
    """
    Build one LLaDA/MDLM-style masked batch.

    Args:
        clean_ids: LongTensor of shape (B, T), without mask ids.
        mask_token_id: the reserved mask token id, normally tokenizer vocab size.
        eps: lower bound for mask probability, avoiding division by zero.
        generator: optional torch.Generator for deterministic tests/sampling.

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
    mask = torch.rand((B, T), device=device, generator=generator) < mask_prob

    # Keep every row trainable even for tiny smoke-test batches.
    empty_rows = ~mask.any(dim=1)
    if empty_rows.any():
        fallback_pos = torch.randint(T, (int(empty_rows.sum().item()),), device=device, generator=generator)
        mask[empty_rows, fallback_pos] = True

    input_ids = clean_ids.clone()
    input_ids[mask] = mask_token_id
    targets = torch.full_like(clean_ids, -1)
    targets[mask] = clean_ids[mask]
    return DiffusionBatch(input_ids=input_ids, targets=targets, mask=mask, mask_prob=mask_prob)


def masked_diffusion_loss(model, clean_ids, mask_token_id, eps=1e-3, generator=None):
    """
    Compute the continuous-time masked diffusion objective.

    The per-token CE is divided by the row's mask probability, matching the
    simple LLaDA/MDLM estimator. The final loss is normalized by B*T.
    """
    batch = make_masked_batch(clean_ids, mask_token_id, eps=eps, generator=generator)
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

    for step in range(steps):
        remaining = (ids == mask_token_id) & editable
        if not remaining.any():
            break

        logits = model(ids)
        logits[..., mask_token_id] = -float("inf")
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
