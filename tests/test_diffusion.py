import torch

from nanochat.diffusion import (
    get_diffusion_vocab_size,
    get_mask_token_id,
    make_masked_batch,
    masked_diffusion_loss,
    sample_masked_diffusion,
)
from nanochat.gpt import GPT, GPTConfig


class TinyTokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

    def get_vocab_size(self):
        return self.vocab_size


def build_tiny_bidirectional_model(vocab_size=17, sequence_len=8):
    config = GPTConfig(
        sequence_len=sequence_len,
        vocab_size=vocab_size,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=32,
        window_pattern="L",
        attention_mode="bidirectional",
    )
    model = GPT(config, pad_vocab_size_to=1)
    model.init_weights()
    return model


def test_mask_token_extends_tokenizer_vocab():
    tokenizer = TinyTokenizer(vocab_size=16)
    assert get_mask_token_id(tokenizer) == 16
    assert get_diffusion_vocab_size(tokenizer) == 17


def test_make_masked_batch_targets_only_masked_positions():
    clean = torch.arange(12, dtype=torch.long).view(2, 6)
    generator = torch.Generator(device=clean.device).manual_seed(123)
    batch = make_masked_batch(clean, mask_token_id=99, eps=0.2, generator=generator)

    assert batch.input_ids.shape == clean.shape
    assert batch.targets.shape == clean.shape
    assert batch.mask.shape == clean.shape
    assert batch.mask.any(dim=1).all()
    assert torch.equal(batch.input_ids[batch.mask], torch.full_like(batch.input_ids[batch.mask], 99))
    assert torch.equal(batch.targets[batch.mask], clean[batch.mask])
    assert torch.equal(batch.targets[~batch.mask], torch.full_like(batch.targets[~batch.mask], -1))


def test_bidirectional_gpt_diffusion_loss_backward():
    model = build_tiny_bidirectional_model()
    clean = torch.randint(0, 16, (2, 8), dtype=torch.long)
    generator = torch.Generator(device=clean.device).manual_seed(123)

    loss, metrics = masked_diffusion_loss(model, clean, mask_token_id=16, eps=0.1, generator=generator)
    loss.backward()

    assert loss.isfinite()
    assert metrics["mask_fraction"] > 0
    assert model.transformer.wte.weight.grad is not None


def test_sample_masked_diffusion_keeps_prompt_and_removes_masks():
    model = build_tiny_bidirectional_model()
    sample = sample_masked_diffusion(
        model,
        mask_token_id=16,
        length=6,
        prompt_tokens=[1, 2],
        steps=4,
        temperature=0.0,
    )

    assert sample[:2] == [1, 2]
    assert len(sample) == 6
    assert 16 not in sample
