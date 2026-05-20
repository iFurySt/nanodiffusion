"""
Generate fixed-prompt samples for a NanoDiffusion checkpoint.

This is a small reporting helper for speedruns. It loads one checkpoint, runs a
few explainable sampling recipes, and writes a markdown table that can be
compared across checkpoints.
"""

import argparse
from pathlib import Path

from nanochat.checkpoint_manager import load_model
from nanochat.common import autodetect_device_type, compute_cleanup, compute_init
from nanochat.diffusion import get_mask_token_id, sample_masked_diffusion


DEFAULT_PROMPTS = [
    "The capital of France is",
    "Once upon a time",
    "In a shocking finding, scientists discovered",
    "The meaning of life is",
    "def fibonacci(n):",
]

SAMPLE_RECIPES = [
    {"name": "greedy", "temperature": 0.0, "top_k": None, "repeat_penalty": 0.0, "no_repeat_ngram_size": 0, "block_size": 0, "steps_scale": 1.0, "remask_low_confidence": False},
    {"name": "temp0.7_top50", "temperature": 0.7, "top_k": 50, "repeat_penalty": 0.0, "no_repeat_ngram_size": 0, "block_size": 0, "steps_scale": 1.0, "remask_low_confidence": False},
    {"name": "temp0.8_top50_repeat0.5", "temperature": 0.8, "top_k": 50, "repeat_penalty": 0.5, "no_repeat_ngram_size": 0, "block_size": 0, "steps_scale": 1.0, "remask_low_confidence": False},
    {"name": "temp0.8_top50_no_repeat3", "temperature": 0.8, "top_k": 50, "repeat_penalty": 0.0, "no_repeat_ngram_size": 3, "block_size": 0, "steps_scale": 1.0, "remask_low_confidence": False},
    {"name": "remask_no_repeat3", "temperature": 0.8, "top_k": 50, "repeat_penalty": 0.0, "no_repeat_ngram_size": 3, "block_size": 0, "steps_scale": 1.0, "remask_low_confidence": True},
    {"name": "block4_no_repeat3", "temperature": 0.8, "top_k": 50, "repeat_penalty": 0.0, "no_repeat_ngram_size": 3, "block_size": 4, "steps_scale": 1.0, "remask_low_confidence": False},
    {"name": "block8_no_repeat3", "temperature": 0.8, "top_k": 50, "repeat_penalty": 0.0, "no_repeat_ngram_size": 3, "block_size": 8, "steps_scale": 1.0, "remask_low_confidence": False},
    {"name": "block16_no_repeat3", "temperature": 0.8, "top_k": 50, "repeat_penalty": 0.0, "no_repeat_ngram_size": 3, "block_size": 16, "steps_scale": 1.0, "remask_low_confidence": False},
    {"name": "half_steps_repeat0.5_no_repeat3", "temperature": 0.8, "top_k": 50, "repeat_penalty": 0.5, "no_repeat_ngram_size": 3, "block_size": 0, "steps_scale": 0.5, "remask_low_confidence": False},
]


def get_forbidden_sample_tokens(tokenizer):
    token_ids = []
    for token in tokenizer.get_special_tokens():
        try:
            token_ids.append(tokenizer.encode_special(token))
        except Exception:
            pass
    return token_ids


def parse_args():
    parser = argparse.ArgumentParser(description="Write a fixed-prompt diffusion sampling report")
    parser.add_argument("-i", "--source", type=str, default="diffusion", choices=["diffusion", "diffusion_sft"])
    parser.add_argument("-g", "--model-tag", type=str, default=None)
    parser.add_argument("-s", "--step", type=int, default=None)
    parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
    parser.add_argument("--prompt", action="append", default=None, help="prompt to sample; can be repeated")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--append", action="store_true")
    return parser.parse_args()


def render_sample(model, tokenizer, prompt, args, recipe, mask_token_id, forbidden_token_ids):
    prompt_tokens = tokenizer(prompt, prepend="<|bos|>")
    length = min(model.config.sequence_len, len(prompt_tokens) + args.max_tokens)
    gen_tokens = max(1, length - len(prompt_tokens))
    steps = max(1, round(gen_tokens * recipe["steps_scale"]))
    ids = sample_masked_diffusion(
        model,
        mask_token_id=mask_token_id,
        length=length,
        prompt_tokens=prompt_tokens,
        steps=steps,
        temperature=recipe["temperature"],
        top_k=recipe["top_k"],
        seed=args.seed,
        forbidden_token_ids=forbidden_token_ids,
        repeat_penalty=recipe["repeat_penalty"],
        no_repeat_ngram_size=recipe["no_repeat_ngram_size"],
        block_size=recipe["block_size"],
        remask_low_confidence=recipe["remask_low_confidence"],
    )
    answer = tokenizer.decode([tok for tok in ids[len(prompt_tokens):] if tok != mask_token_id])
    return prompt + answer


def build_report(args, model, tokenizer, meta):
    mask_token_id = meta.get("mask_token_id", get_mask_token_id(tokenizer))
    prompts = args.prompt if args.prompt else DEFAULT_PROMPTS
    forbidden_token_ids = get_forbidden_sample_tokens(tokenizer)
    step = args.step if args.step is not None else meta.get("step", "latest")
    lines = [
        f"## Diffusion Samples: {args.source}/{args.model_tag or 'default'} step {step}",
        "",
        f"- source: `{args.source}`",
        f"- model_tag: `{args.model_tag or 'default'}`",
        f"- step: `{step}`",
        f"- max_tokens: `{args.max_tokens}`",
        f"- seed: `{args.seed}`",
        "",
    ]
    for prompt in prompts:
        lines.extend([f"### `{prompt}`", ""])
        for recipe in SAMPLE_RECIPES:
            sample = render_sample(model, tokenizer, prompt, args, recipe, mask_token_id, forbidden_token_ids)
            lines.extend(
                [
                    f"**{recipe['name']}** "
                    f"(temp={recipe['temperature']}, top_k={recipe['top_k']}, "
                    f"repeat_penalty={recipe['repeat_penalty']}, "
                    f"no_repeat_ngram_size={recipe['no_repeat_ngram_size']}, "
                    f"block_size={recipe['block_size']}, "
                    f"steps_scale={recipe['steps_scale']}, "
                    f"remask_low_confidence={recipe['remask_low_confidence']})",
                    "",
                    "```text",
                    sample,
                    "```",
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def main():
    args = parse_args()
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    _ddp, _rank, _local_rank, _world_size, device = compute_init(device_type)
    model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)
    report = build_report(args, model, tokenizer, meta)

    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if args.append else "w"
        with path.open(mode, encoding="utf-8") as f:
            if args.append and path.stat().st_size > 0:
                f.write("\n")
            f.write(report)
    else:
        print(report, end="")
    compute_cleanup()


if __name__ == "__main__":
    main()
