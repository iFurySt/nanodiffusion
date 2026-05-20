"""
Evaluate and sample from a masked diffusion base model.

Examples:

    python -m scripts.diffusion_base_eval --help

    python -m scripts.diffusion_base_eval --model-tag=diffusion_d20 --eval=loss,sample
"""

import argparse

import torch
import torch.distributed as dist

from nanochat.checkpoint_manager import load_model
from nanochat.common import autodetect_device_type, compute_cleanup, compute_init, is_ddp_initialized, print0
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.diffusion import get_mask_token_id, masked_diffusion_loss, sample_masked_diffusion


def get_forbidden_sample_tokens(tokenizer):
    token_ids = []
    for token in tokenizer.get_special_tokens():
        try:
            token_ids.append(tokenizer.encode_special(token))
        except Exception:
            pass
    return token_ids


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a masked diffusion language model")
    parser.add_argument("--eval", type=str, default="loss,sample", help="comma-separated: loss,sample")
    parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
    parser.add_argument("--model-tag", type=str, default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--device-batch-size", type=int, default=16)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--mask-eps", type=float, default=1e-3)
    parser.add_argument("--mask-max-prob", type=float, default=1.0)
    parser.add_argument("--no-mask-loss-reweight", action="store_true")
    parser.add_argument("--prompt", type=str, default="The capital of France is")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--repeat-penalty", type=float, default=0.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--block-size", type=int, default=0, help="generate answer in fixed blocks; 0 disables")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


@torch.inference_mode()
def evaluate_loss(model, tokenizer, device, args, mask_token_id, ddp_world_size):
    loader = tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer,
        args.device_batch_size,
        model.config.sequence_len,
        split="val",
        device=device,
    )
    total_loss = torch.tensor(0.0, device=device)
    total_batches = 0
    for _ in range(args.eval_batches):
        clean_ids, _targets = next(loader)
        loss, _metrics = masked_diffusion_loss(
            model,
            clean_ids,
            mask_token_id,
            eps=args.mask_eps,
            max_mask_prob=args.mask_max_prob,
            loss_reweight=not args.no_mask_loss_reweight,
        )
        total_loss += loss
        total_batches += 1
    if is_ddp_initialized():
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        total_batches *= ddp_world_size
    return (total_loss / total_batches).item()


def run_sample(model, tokenizer, args, mask_token_id):
    prompt_tokens = tokenizer(args.prompt, prepend="<|bos|>")
    length = min(model.config.sequence_len, len(prompt_tokens) + args.max_tokens)
    ids = sample_masked_diffusion(
        model,
        mask_token_id=mask_token_id,
        length=length,
        prompt_tokens=prompt_tokens,
        steps=args.steps,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
        forbidden_token_ids=get_forbidden_sample_tokens(tokenizer),
        repeat_penalty=args.repeat_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        block_size=args.block_size,
    )
    prompt_len = len(prompt_tokens)
    return args.prompt + tokenizer.decode([tok for tok in ids[prompt_len:] if tok != mask_token_id])


def main():
    args = parse_args()
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    _ddp, ddp_rank, _ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    eval_modes = {item.strip() for item in args.eval.split(",") if item.strip()}

    model, tokenizer, meta_data = load_model("diffusion", device, phase="eval", model_tag=args.model_tag, step=args.step)
    mask_token_id = meta_data.get("mask_token_id", get_mask_token_id(tokenizer))
    print0(f"Loaded diffusion model with mask_token_id={mask_token_id}")

    if "loss" in eval_modes:
        val_loss = evaluate_loss(model, tokenizer, device, args, mask_token_id, ddp_world_size)
        print0(f"Validation diffusion loss: {val_loss:.6f}")

    if "sample" in eval_modes and ddp_rank == 0:
        sample = run_sample(model, tokenizer, args, mask_token_id)
        print0(sample)

    compute_cleanup()


if __name__ == "__main__":
    main()
