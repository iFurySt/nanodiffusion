"""
Pretrain a masked diffusion base model.

Examples:

    python -m scripts.diffusion_base_train --help

    # CPU smoke test once dataset/tokenizer are prepared:
    python -m scripts.diffusion_base_train \
      --device-type=cpu --depth=2 --aspect-ratio=16 --head-dim=16 \
      --max-seq-len=64 --device-batch-size=2 --num-iterations=5 \
      --total-batch-size=128 --eval-every=-1

    # 8xA100:
    torchrun --nproc_per_node=8 -m scripts.diffusion_base_train
"""

import argparse
import gc
import json
import math
import os
import time
from dataclasses import asdict

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.distributed as dist

from nanochat.checkpoint_manager import load_checkpoint, save_checkpoint
from nanochat.common import (
    COMPUTE_DTYPE,
    COMPUTE_DTYPE_REASON,
    autodetect_device_type,
    compute_cleanup,
    compute_init,
    get_base_dir,
    get_dist_info,
    get_peak_flops,
    is_ddp_initialized,
    print0,
    print_banner,
)
from nanochat.dataloader import (
    tokenizing_distributed_data_loader_bos_bestfit,
    tokenizing_distributed_data_loader_with_state_bos_bestfit,
)
from nanochat.diffusion import get_diffusion_vocab_size, get_mask_token_id, masked_diffusion_loss
from nanochat.flash_attention import HAS_FA3, USE_FA3
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain a masked diffusion language model")
    # Runtime
    parser.add_argument("--run", type=str, default="dummy", help="run name for logs")
    parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
    parser.add_argument("--compile", action="store_true", help="torch.compile the model")
    # Model architecture
    parser.add_argument("--depth", type=int, default=20)
    parser.add_argument("--aspect-ratio", type=int, default=64, help="model_dim = depth * aspect_ratio")
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    # Optimization
    parser.add_argument("--num-iterations", type=int, default=-1)
    parser.add_argument("--target-param-data-ratio", type=float, default=12)
    parser.add_argument("--device-batch-size", type=int, default=32)
    parser.add_argument("--total-batch-size", type=int, default=-1, help="total tokens per optimizer step")
    parser.add_argument("--embedding-lr", type=float, default=0.3)
    parser.add_argument("--unembedding-lr", type=float, default=0.008)
    parser.add_argument("--matrix-lr", type=float, default=0.02)
    parser.add_argument("--scalar-lr", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=0.28)
    parser.add_argument("--warmup-steps", type=int, default=40)
    parser.add_argument("--warmdown-ratio", type=float, default=0.65)
    parser.add_argument("--final-lr-frac", type=float, default=0.05)
    parser.add_argument("--mask-eps", type=float, default=1e-3)
    parser.add_argument("--mask-max-prob", type=float, default=1.0)
    parser.add_argument("--no-mask-loss-reweight", action="store_true")
    parser.add_argument("--mask-pattern", type=str, default="full", choices=["full", "suffix", "suffix_all", "suffix_span", "suffix_span_all"])
    parser.add_argument("--prefix-min-frac", type=float, default=0.25)
    parser.add_argument("--prefix-max-frac", type=float, default=0.75)
    parser.add_argument("--span-tokens", type=int, default=128)
    parser.add_argument("--loss-normalization", type=str, default="all", choices=["all", "eligible"])
    parser.add_argument("--mask-sampling", type=str, default="uniform", choices=["uniform", "antithetic"])
    parser.add_argument("--resume-from-step", type=int, default=-1)
    # Evaluation / output
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=-1)
    parser.add_argument("--model-tag", type=str, default=None)
    return parser.parse_args()


def build_model_meta(args, vocab_size):
    base_dim = args.depth * args.aspect_ratio
    model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
    num_heads = model_dim // args.head_dim
    config = GPTConfig(
        sequence_len=args.max_seq_len,
        vocab_size=vocab_size,
        n_layer=args.depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
        window_pattern="L",
        attention_mode="bidirectional",
    )
    with torch.device("meta"):
        model = GPT(config)
    return model


@torch.inference_mode()
def evaluate_diffusion_loss(model, tokenizer, device, args, mask_token_id, ddp_world_size):
    model.eval()
    loader = tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer,
        args.device_batch_size,
        args.max_seq_len,
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
            mask_pattern=args.mask_pattern,
            min_prefix_frac=args.prefix_min_frac,
            max_prefix_frac=args.prefix_max_frac,
            span_tokens=args.span_tokens,
            loss_normalization=args.loss_normalization,
            mask_sampling=args.mask_sampling,
        )
        total_loss += loss
        total_batches += 1
    if is_ddp_initialized():
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        total_batches *= ddp_world_size
    model.train()
    return (total_loss / total_batches).item()


def main():
    args = parse_args()
    user_config = vars(args).copy()
    print_banner()

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    master_process = ddp_rank == 0
    synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
    get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

    if device_type == "cuda":
        gpu_device_name = torch.cuda.get_device_name(0)
        gpu_peak_flops = get_peak_flops(gpu_device_name)
        print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
    else:
        gpu_peak_flops = float("inf")
    print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")
    print0(f"Flash attention: {'FA3' if USE_FA3 else 'SDPA fallback'}")
    if HAS_FA3 and not USE_FA3:
        print0("FA3 is installed but not active for this dtype/device")

    tokenizer = get_tokenizer()
    tokenizer_vocab_size = tokenizer.get_vocab_size()
    mask_token_id = get_mask_token_id(tokenizer)
    diffusion_vocab_size = get_diffusion_vocab_size(tokenizer)
    print0(f"Tokenizer vocab size: {tokenizer_vocab_size:,}")
    print0(f"Mask token id: {mask_token_id:,}")
    print0(f"Diffusion model vocab size: {diffusion_vocab_size:,}")

    model = build_model_meta(args, diffusion_vocab_size)
    model_config = model.config
    model_config_kwargs = asdict(model_config)
    print0(f"Model config:\n{json.dumps(model_config_kwargs, indent=2)}")
    model.to_empty(device=device)
    model.init_weights()

    base_dir = get_base_dir()
    output_dirname = args.model_tag if args.model_tag else f"diffusion_d{args.depth}"
    checkpoint_dir = os.path.join(base_dir, "diffusion_checkpoints", output_dirname)

    resuming = args.resume_from_step != -1
    if resuming:
        print0(f"Resuming optimization from step {args.resume_from_step}")
        model_data, optimizer_data, meta_data = load_checkpoint(
            checkpoint_dir,
            args.resume_from_step,
            device,
            load_optimizer=True,
            rank=ddp_rank,
        )
        model.load_state_dict(model_data, strict=True, assign=True)

    orig_model = model
    train_model = torch.compile(model, dynamic=False) if args.compile else model

    param_counts = orig_model.num_scaling_params()
    num_params = param_counts["total"]
    num_scaling_params = param_counts["transformer_matrices"] + param_counts["lm_head"]
    num_flops_per_token = orig_model.estimate_flops()
    print0(f"Total parameters: {num_params:,}")
    print0(f"Scaling parameters: {num_scaling_params:,}")
    print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

    total_batch_size = args.total_batch_size
    if total_batch_size == -1:
        total_batch_size = 2**19
        print0(f"Using default total batch size: {total_batch_size:,} tokens")
    if args.num_iterations > 0:
        num_iterations = args.num_iterations
    else:
        target_tokens = int(args.target_param_data_ratio * num_scaling_params)
        num_iterations = max(1, target_tokens // total_batch_size)
    print0(f"Training iterations: {num_iterations:,}")

    optimizer = orig_model.setup_optimizer(
        unembedding_lr=args.unembedding_lr,
        embedding_lr=args.embedding_lr,
        matrix_lr=args.matrix_lr,
        scalar_lr=args.scalar_lr,
        weight_decay=args.weight_decay,
    )
    if resuming:
        optimizer.load_state_dict(optimizer_data)

    dataloader_resume_state_dict = None if not resuming else meta_data["dataloader_state_dict"]
    train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
        tokenizer,
        args.device_batch_size,
        args.max_seq_len,
        split="train",
        device=device,
        resume_state_dict=dataloader_resume_state_dict,
    )
    clean_ids, _next_token_targets, dataloader_state_dict = next(train_loader)

    tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len
    world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
    assert total_batch_size % world_tokens_per_fwdbwd == 0
    grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
    print0(f"Tokens / micro-batch / rank: {tokens_per_fwdbwd:,}")
    print0(f"Gradient accumulation steps: {grad_accum_steps}")

    def get_lr_multiplier(step):
        warmup = max(1, args.warmup_steps)
        warmdown = round(args.warmdown_ratio * num_iterations)
        if step < warmup:
            return (step + 1) / warmup
        if warmdown <= 0 or step <= num_iterations - warmdown:
            return 1.0
        progress = (num_iterations - step) / warmdown
        return progress + (1 - progress) * args.final_lr_frac

    if not resuming:
        step = 0
        smooth_train_loss = 0.0
        min_val_loss = float("inf")
        total_training_time = 0.0
    else:
        step = meta_data["step"]
        loop_state = meta_data["loop_state"]
        smooth_train_loss = loop_state["smooth_train_loss"]
        min_val_loss = loop_state["min_val_loss"]
        total_training_time = loop_state["total_training_time"]

    train_model.train()
    while True:
        last_step = step == num_iterations

        if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
            val_loss = evaluate_diffusion_loss(orig_model, tokenizer, device, args, mask_token_id, ddp_world_size)
            min_val_loss = min(min_val_loss, val_loss)
            print0(f"Step {step:05d} | validation diffusion loss: {val_loss:.6f}")

        should_save = last_step or (step > 0 and args.save_every > 0 and step % args.save_every == 0)
        if should_save:
            save_checkpoint(
                checkpoint_dir,
                step,
                orig_model.state_dict(),
                optimizer.state_dict(),
                {
                    "step": step,
                    "model_config": model_config_kwargs,
                    "user_config": user_config,
                    "mask_token_id": mask_token_id,
                    "tokenizer_vocab_size": tokenizer_vocab_size,
                    "device_batch_size": args.device_batch_size,
                    "max_seq_len": args.max_seq_len,
                    "total_batch_size": total_batch_size,
                    "dataloader_state_dict": dataloader_state_dict,
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
        for micro_step in range(grad_accum_steps):
            loss, metrics = masked_diffusion_loss(
                train_model,
                clean_ids,
                mask_token_id,
                eps=args.mask_eps,
                max_mask_prob=args.mask_max_prob,
                loss_reweight=not args.no_mask_loss_reweight,
                mask_pattern=args.mask_pattern,
                min_prefix_frac=args.prefix_min_frac,
                max_prefix_frac=args.prefix_max_frac,
                span_tokens=args.span_tokens,
                loss_normalization=args.loss_normalization,
                mask_sampling=args.mask_sampling,
            )
            train_loss = loss.detach()
            (loss / grad_accum_steps).backward()
            clean_ids, _next_token_targets, dataloader_state_dict = next(train_loader)

        lrm = get_lr_multiplier(step)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
        optimizer.step()
        train_model.zero_grad(set_to_none=True)

        synchronize()
        dt = time.time() - t0
        train_loss_f = train_loss.item()
        smooth_train_loss = 0.9 * smooth_train_loss + 0.1 * train_loss_f
        debiased = smooth_train_loss / (1 - 0.9 ** (step + 1))
        if step > 10:
            total_training_time += dt
        tok_per_sec = int(total_batch_size / dt)
        flops_per_sec = num_flops_per_token * total_batch_size / dt
        mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
        epoch = f"{dataloader_state_dict['epoch']} pq: {dataloader_state_dict['pq_idx']} rg: {dataloader_state_dict['rg_idx']}"
        print0(
            f"step {step:05d}/{num_iterations:05d} | loss: {debiased:.6f} | "
            f"mask: {metrics['mask_fraction'].item():.3f} | lrm: {lrm:.2f} | "
            f"dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | "
            f"bf16_mfu: {mfu:.2f} | epoch: {epoch}"
        )

        step += 1
        if step == 1:
            gc.collect()
            gc.freeze()
            gc.disable()
        elif step % 5000 == 0:
            gc.collect()

    print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
    print0(f"Total training time: {total_training_time/60:.2f}m")
    print0(f"Minimum validation diffusion loss: {min_val_loss:.6f}")

    from nanochat.report import get_report
    get_report().log(section="Diffusion base model training", data=[
        user_config,
        {
            "Number of parameters": num_params,
            "Number of FLOPs per token": f"{num_flops_per_token:e}",
            "Calculated number of iterations": num_iterations,
            "Total batch size": total_batch_size,
            "Mask token id": mask_token_id,
            "Mask max probability": args.mask_max_prob,
            "Mask loss reweight": not args.no_mask_loss_reweight,
            "Mask pattern": args.mask_pattern,
            "Prefix min fraction": args.prefix_min_frac,
            "Prefix max fraction": args.prefix_max_frac,
            "Span tokens": args.span_tokens,
            "Loss normalization": args.loss_normalization,
            "Mask sampling": args.mask_sampling,
        },
        {
            "Minimum validation diffusion loss": min_val_loss,
            "Total training time": total_training_time,
        },
    ])

    if master_process and args.run != "dummy":
        print0(f"Run name: {args.run}")
    compute_cleanup()


if __name__ == "__main__":
    main()
