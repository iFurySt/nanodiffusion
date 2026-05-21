"""
Minimal CLI for sampling from a NanoDiffusion checkpoint.

This is not an autoregressive chat state machine. It keeps the prompt fixed and
fills a fixed-size answer window by iterative masked denoising.
"""

import argparse

from nanochat.checkpoint_manager import load_model
from nanochat.common import autodetect_device_type, compute_cleanup, compute_init
from nanochat.diffusion import get_mask_token_id, sample_masked_diffusion


def get_forbidden_sample_tokens(tokenizer):
    token_ids = []
    for token in tokenizer.get_special_tokens():
        try:
            token_ids.append(tokenizer.encode_special(token))
        except Exception:
            pass
    return token_ids


def parse_args():
    parser = argparse.ArgumentParser(description="Sample from a NanoDiffusion model")
    parser.add_argument("-i", "--source", type=str, default="diffusion_sft", choices=["diffusion", "diffusion_sft"])
    parser.add_argument("-g", "--model-tag", type=str, default=None)
    parser.add_argument("-s", "--step", type=int, default=None)
    parser.add_argument("-p", "--prompt", type=str, default="")
    parser.add_argument("--max-tokens", type=int, default=64, help="number of tokens to denoise after the prompt")
    parser.add_argument("--steps", type=int, default=None, help="denoising steps (default: answer length)")
    parser.add_argument("-t", "--temperature", type=float, default=0.0)
    parser.add_argument("-k", "--top-k", type=int, default=None)
    parser.add_argument("--repeat-penalty", type=float, default=0.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--block-size", type=int, default=0, help="generate answer in fixed blocks; 0 disables")
    parser.add_argument("--remask-low-confidence", action="store_true")
    parser.add_argument("--cfg-scale", type=float, default=0.0)
    parser.add_argument("--reveal-strategy", type=str, default="confidence", choices=["confidence", "left_to_right"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device-type", type=str, default="", choices=["cuda", "cpu", "mps"])
    return parser.parse_args()


def render(model, tokenizer, prompt, args, mask_token_id):
    prompt_tokens = tokenizer(prompt, prepend="<|bos|>")
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
        remask_low_confidence=args.remask_low_confidence,
        cfg_scale=args.cfg_scale,
        reveal_strategy=args.reveal_strategy,
    )
    answer_ids = [tok for tok in ids[len(prompt_tokens):] if tok != mask_token_id]
    return prompt + tokenizer.decode(answer_ids)


def main():
    args = parse_args()
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    _ddp, _rank, _local_rank, _world_size, device = compute_init(device_type)
    model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)
    mask_token_id = meta.get("mask_token_id", get_mask_token_id(tokenizer))

    if args.prompt:
        print(render(model, tokenizer, args.prompt, args, mask_token_id))
        compute_cleanup()
        return

    print("\nNanoDiffusion Interactive Mode")
    print("-" * 50)
    print("Type 'quit' or 'exit' to end")
    print("-" * 50)
    while True:
        try:
            prompt = input("\nPrompt: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if prompt.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break
        if not prompt:
            continue
        print(render(model, tokenizer, prompt, args, mask_token_id))

    compute_cleanup()


if __name__ == "__main__":
    main()
