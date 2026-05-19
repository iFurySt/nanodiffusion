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


def test_make_masked_batch_respects_eligible_mask():
    clean = torch.arange(12, dtype=torch.long).view(2, 6)
    eligible = torch.zeros_like(clean, dtype=torch.bool)
    eligible[:, 3:] = True
    generator = torch.Generator(device=clean.device).manual_seed(123)
    batch = make_masked_batch(clean, mask_token_id=99, eps=0.9, generator=generator, eligible_mask=eligible)

    assert not batch.mask[:, :3].any()
    assert batch.mask[:, 3:].any(dim=1).all()
    assert torch.equal(batch.targets[:, :3], torch.full_like(batch.targets[:, :3], -1))


def test_bidirectional_gpt_diffusion_loss_backward():
    model = build_tiny_bidirectional_model()
    clean = torch.randint(0, 16, (2, 8), dtype=torch.long)
    generator = torch.Generator(device=clean.device).manual_seed(123)

    loss, metrics = masked_diffusion_loss(model, clean, mask_token_id=16, eps=0.1, generator=generator)
    loss.backward()

    assert loss.isfinite()
    assert metrics["mask_fraction"] > 0
    assert model.transformer.wte.weight.grad is not None


def test_diffusion_loss_can_train_answer_span_only():
    model = build_tiny_bidirectional_model()
    clean = torch.randint(0, 16, (2, 8), dtype=torch.long)
    answer_mask = torch.zeros_like(clean, dtype=torch.bool)
    answer_mask[:, 4:] = True
    generator = torch.Generator(device=clean.device).manual_seed(123)

    batch = make_masked_batch(clean, mask_token_id=16, eps=0.9, generator=generator, eligible_mask=answer_mask)
    assert not batch.mask[:, :4].any()
    loss, metrics = masked_diffusion_loss(
        model,
        clean,
        mask_token_id=16,
        eps=0.1,
        generator=generator,
        eligible_mask=answer_mask,
    )
    loss.backward()

    assert loss.isfinite()
    assert metrics["mask_fraction"] > 0


def test_tiny_embedding_model_supported_for_cpu_smoke():
    small_config = GPTConfig(
        sequence_len=8,
        vocab_size=17,
        n_layer=1,
        n_head=1,
        n_kv_head=1,
        n_embd=16,
        window_pattern="L",
        attention_mode="bidirectional",
    )
    model = GPT(small_config, pad_vocab_size_to=1)
    model.init_weights()
    clean = torch.randint(0, 16, (1, 8), dtype=torch.long)
    loss, _metrics = masked_diffusion_loss(model, clean, mask_token_id=16, eps=0.1)
    loss.backward()
    assert loss.isfinite()


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


def test_sample_masked_diffusion_respects_forbidden_tokens():
    model = build_tiny_bidirectional_model()
    sample = sample_masked_diffusion(
        model,
        mask_token_id=16,
        length=6,
        prompt_tokens=[1, 2],
        steps=4,
        temperature=0.0,
        forbidden_token_ids=[0, 3, 4, 5],
    )

    assert sample[:2] == [1, 2]
    assert not ({0, 3, 4, 5, 16} & set(sample[2:]))
