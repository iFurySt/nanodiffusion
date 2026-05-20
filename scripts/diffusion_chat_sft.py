"""
Supervised fine-tuning for NanoDiffusion.

The prompt side of each conversation stays fixed. Only assistant-answer tokens
are eligible for masking and loss.

Examples:

    python -m scripts.diffusion_chat_sft --help

    python -m scripts.diffusion_chat_sft \
      --model-tag=diffusion_d20 \
      --data-jsonl=$NANODIFFUSION_BASE_DIR/identity_conversations.jsonl
"""

import argparse
import gc
import json
import os
import time
from dataclasses import asdict

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.distributed as dist

from nanochat.checkpoint_manager import load_model, load_optimizer_state, save_checkpoint
from nanochat.common import (
    COMPUTE_DTYPE,
    COMPUTE_DTYPE_REASON,
    autodetect_device_type,
    compute_cleanup,
    compute_init,
    get_base_dir,
    get_peak_flops,
    is_ddp_initialized,
    print0,
)
from nanochat.diffusion import get_mask_token_id, masked_diffusion_loss
from tasks.common import TaskMixture
from tasks.customjson import CustomJSON


def parse_args():
    parser = argparse.ArgumentParser(description="SFT a masked diffusion language model")
    parser.add_argument("--run", type=str, default="dummy")
    parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--model-tag", type=str, default=None, help="diffusion base model tag to load")
    parser.add_argument("--model-step", type=int, default=None)
    parser.add_argument("--load-optimizer", type=int, default=0)
    parser.add_argument("--data-jsonl", type=str, default=None, help="JSONL conversations for SFT")
    parser.add_argument("--include-smoltalk", action="store_true", help="also train on SmolTalk train split")
    parser.add_argument("--num-iterations", type=int, default=100)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--device-batch-size", type=int, default=4)
    parser.add_argument("--total-batch-size", type=int, default=8192)
    parser.add_argument("--embedding-lr", type=float, default=0.03)
    parser.add_argument("--unembedding-lr", type=float, default=0.001)
    parser.add_argument("--matrix-lr", type=float, default=0.002)
    parser.add_argument("--scalar-lr", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--warmdown-ratio", type=float, default=0.5)
    parser.add_argument("--final-lr-frac", type=float, default=0.0)
    parser.add_argument("--mask-eps", type=float, default=1e-3)
    parser.add_argument("--eval-every", type=int, default=-1)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=-1)
    parser.add_argument("--output-tag", type=str, default=None)
    return parser.parse_args()


def build_dataset(args):
    base_dir = get_base_dir()
    data_jsonl = args.data_jsonl or os.path.join(base_dir, "identity_conversations.jsonl")
    tasks = []
    if data_jsonl:
        if not os.path.exists(data_jsonl):
            raise FileNotFoundError(
                f"SFT data not found: {data_jsonl}. Pass --data-jsonl or download identity_conversations.jsonl."
            )
        tasks.append(CustomJSON(filepath=data_jsonl))
    if args.include_smoltalk:
        from tasks.smoltalk import SmolTalk

        tasks.append(SmolTalk(split="train"))
    if not tasks:
        raise ValueError("No SFT data selected")
    dataset = TaskMixture(tasks)
    assert len(dataset) > 0, "SFT dataset is empty"
    return dataset


def sft_loader(dataset, tokenizer, args, device, ddp_rank, ddp_world_size):
    """Yield clean ids and eligible assistant-token masks."""
    row_capacity = args.max_seq_len
    bos_token = tokenizer.get_bos_token_id()
    dataset_size = len(dataset)
    cursor = ddp_rank % dataset_size
    use_cuda = device.type == "cuda"

    while True:
        rows = []
        eligible_rows = []
        for _ in range(args.device_batch_size):
            for _attempt in range(dataset_size):
                conversation = dataset[cursor]
                ids, mask = tokenizer.render_conversation(conversation, max_tokens=row_capacity)
                cursor += ddp_world_size
                if cursor >= dataset_size:
                    cursor = cursor % dataset_size
                # If truncation removed the assistant answer, this example has no
                # trainable diffusion target. Skip instead of silently training
                # a zero-loss batch.
                if any(mask):
                    break
            else:
                raise RuntimeError(
                    "No SFT examples contain assistant tokens after truncation. "
                    "Increase --max-seq-len or inspect the SFT data."
                )

            if len(ids) < row_capacity:
                pad = row_capacity - len(ids)
                ids = ids + [bos_token] * pad
                mask = mask + [0] * pad
            rows.append(ids[:row_capacity])
            eligible_rows.append(mask[:row_capacity])

        clean_ids = torch.tensor(rows, dtype=torch.long, pin_memory=use_cuda).to(device=device, non_blocking=use_cuda)
        eligible_mask = torch.tensor(eligible_rows, dtype=torch.bool, pin_memory=use_cuda).to(device=device, non_blocking=use_cuda)
        yield clean_ids.contiguous(), eligible_mask.contiguous()


@torch.inference_mode()
def evaluate_sft_loss(model, dataset, tokenizer, args, device, mask_token_id, ddp_rank, ddp_world_size):
    loader = sft_loader(dataset, tokenizer, args, device, ddp_rank, ddp_world_size)
    total_loss = torch.tensor(0.0, device=device)
    total_batches = 0
    model.eval()
    for _ in range(args.eval_batches):
        clean_ids, eligible_mask = next(loader)
        loss, _metrics = masked_diffusion_loss(
            model,
            clean_ids,
            mask_token_id,
            eps=args.mask_eps,
            eligible_mask=eligible_mask,
        )
        total_loss += loss
        total_batches += 1
    if is_ddp_initialized():
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        total_batches *= ddp_world_size
    model.train()
    return (total_loss / total_batches).item()


def lr_multiplier(progress, args):
    if args.warmup_ratio > 0 and progress < args.warmup_ratio:
        return progress / args.warmup_ratio
    if args.warmdown_ratio > 0 and progress > 1.0 - args.warmdown_ratio:
        decay = (progress - (1.0 - args.warmdown_ratio)) / args.warmdown_ratio
        return (1.0 - decay) + decay * args.final_lr_frac
    return 1.0


def main():
    args = parse_args()
    user_config = vars(args).copy()
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, _ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
    get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
    if device_type == "cuda":
        gpu_peak_flops = get_peak_flops(torch.cuda.get_device_name(0))
    else:
        gpu_peak_flops = float("inf")
    print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")

    model, tokenizer, meta = load_model("diffusion", device, phase="train", model_tag=args.model_tag, step=args.model_step)
    if args.max_seq_len is None:
        args.max_seq_len = model.config.sequence_len
    elif args.max_seq_len > model.config.sequence_len:
        print0(f"Extending model config sequence_len from {model.config.sequence_len} to {args.max_seq_len} for SFT")
        model.config.sequence_len = args.max_seq_len
    mask_token_id = meta.get("mask_token_id", get_mask_token_id(tokenizer))
    dataset = build_dataset(args)
    print0(f"SFT dataset rows: {len(dataset):,}")
    print0(f"Mask token id: {mask_token_id}")

    orig_model = model
    train_model = torch.compile(model, dynamic=False) if args.compile else model
    num_flops_per_token = orig_model.estimate_flops()
    tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len
    world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
    assert args.total_batch_size % world_tokens_per_fwdbwd == 0
    grad_accum_steps = args.total_batch_size // world_tokens_per_fwdbwd
    print0(f"Tokens / micro-batch / rank: {tokens_per_fwdbwd:,}")
    print0(f"Gradient accumulation steps: {grad_accum_steps}")

    optimizer = orig_model.setup_optimizer(
        unembedding_lr=args.unembedding_lr,
        embedding_lr=args.embedding_lr,
        matrix_lr=args.matrix_lr,
        scalar_lr=args.scalar_lr,
        weight_decay=args.weight_decay,
    )
    if args.load_optimizer:
        optimizer_data = load_optimizer_state("diffusion", device, rank=ddp_rank, model_tag=args.model_tag, step=args.model_step)
        if optimizer_data is not None:
            base_lrs = [group["lr"] for group in optimizer.param_groups]
            optimizer.load_state_dict(optimizer_data)
            for group, base_lr in zip(optimizer.param_groups, base_lrs):
                group["lr"] = base_lr
                group["initial_lr"] = base_lr

    loader = sft_loader(dataset, tokenizer, args, device, ddp_rank, ddp_world_size)
    clean_ids, eligible_mask = next(loader)
    output_tag = args.output_tag or (args.model_tag if args.model_tag else "diffusion_sft")
    checkpoint_dir = os.path.join(get_base_dir(), "diffusion_sft_checkpoints", output_tag)

    train_model.train()
    smooth_train_loss = 0.0
    min_val_loss = float("inf")
    total_training_time = 0.0
    for step in range(args.num_iterations + 1):
        last_step = step == args.num_iterations
        if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
            val_loss = evaluate_sft_loss(orig_model, dataset, tokenizer, args, device, mask_token_id, ddp_rank, ddp_world_size)
            min_val_loss = min(min_val_loss, val_loss)
            print0(f"Step {step:05d} | validation diffusion SFT loss: {val_loss:.6f}")

        should_save = last_step or (step > 0 and args.save_every > 0 and step % args.save_every == 0)
        if should_save:
            save_checkpoint(
                checkpoint_dir,
                step,
                orig_model.state_dict(),
                optimizer.state_dict(),
                {
                    "step": step,
                    "model_config": asdict(orig_model.config),
                    "user_config": user_config,
                    "mask_token_id": mask_token_id,
                    "tokenizer_vocab_size": tokenizer.get_vocab_size(),
                    "loop_state": {
                        "smooth_train_loss": smooth_train_loss,
                        "min_val_loss": min_val_loss,
                        "total_training_time": total_training_time,
                    },
                },
                rank=ddp_rank,
            )
        if last_step:
            break

        synchronize()
        t0 = time.time()
        for _ in range(grad_accum_steps):
            loss, metrics = masked_diffusion_loss(
                train_model,
                clean_ids,
                mask_token_id,
                eps=args.mask_eps,
                eligible_mask=eligible_mask,
            )
            train_loss = loss.detach()
            (loss / grad_accum_steps).backward()
            clean_ids, eligible_mask = next(loader)

        progress = (step + 1) / args.num_iterations
        lrm = lr_multiplier(progress, args)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
        optimizer.step()
        train_model.zero_grad(set_to_none=True)
        synchronize()

        dt = time.time() - t0
        smooth_train_loss = 0.9 * smooth_train_loss + 0.1 * train_loss.item()
        debiased = smooth_train_loss / (1 - 0.9 ** (step + 1))
        if step > 10:
            total_training_time += dt
        tok_per_sec = int(args.total_batch_size / dt)
        flops_per_sec = num_flops_per_token * args.total_batch_size / dt
        mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
        print0(
            f"step {step:05d}/{args.num_iterations:05d} | loss: {debiased:.6f} | "
            f"mask: {metrics['mask_fraction'].item():.3f} | lrm: {lrm:.2f} | "
            f"dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f}"
        )

        if step == 0:
            gc.collect()
            gc.freeze()
            gc.disable()
        elif step % 5000 == 0:
            gc.collect()

    print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
    print0(f"Total training time: {total_training_time/60:.2f}m")
    print0(f"Minimum validation diffusion SFT loss: {min_val_loss:.6f}")
    compute_cleanup()


if __name__ == "__main__":
    main()
