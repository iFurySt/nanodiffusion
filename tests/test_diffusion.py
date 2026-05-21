import torch
import torch.nn as nn
from types import SimpleNamespace

from nanochat.diffusion import (
    get_diffusion_vocab_size,
    get_mask_token_id,
    make_masked_batch,
    make_suffix_eligible_mask,
    make_suffix_span_masks,
    masked_diffusion_loss,
    sample_masked_diffusion,
)
from nanochat.gpt import GPT, GPTConfig
from scripts.diffusion_chat_sft import sft_loader


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


class FixedLogitModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = GPTConfig(
            sequence_len=4,
            vocab_size=8,
            n_layer=1,
            n_head=1,
            n_kv_head=1,
            n_embd=8,
            window_pattern="L",
            attention_mode="bidirectional",
        )

    def get_device(self):
        return torch.device("cpu")

    def forward(self, ids):
        logits = torch.zeros((*ids.shape, self.config.vocab_size), dtype=torch.float32)
        logits[..., 3] = 3.0
        logits[..., 4] = 2.0
        logits[..., 5] = 1.0
        return logits


class TinyConversationTokenizer:
    def get_bos_token_id(self):
        return 0

    def render_conversation(self, conversation, max_tokens):
        del conversation, max_tokens
        return [1, 2, 3], [0, 1, 1]


class TinyConversationDataset:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


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


def test_make_masked_batch_force_masks_inputs_without_targets():
    clean = torch.arange(12, dtype=torch.long).view(2, 6)
    eligible = torch.zeros_like(clean, dtype=torch.bool)
    eligible[:, 2:4] = True
    force = torch.zeros_like(clean, dtype=torch.bool)
    force[:, 4:] = True
    generator = torch.Generator(device=clean.device).manual_seed(123)
    batch = make_masked_batch(
        clean,
        mask_token_id=99,
        eps=0.9,
        generator=generator,
        eligible_mask=eligible,
        force_mask=force,
    )

    assert torch.equal(batch.input_ids[force], torch.full_like(batch.input_ids[force], 99))
    assert torch.equal(batch.targets[force], torch.full_like(batch.targets[force], -1))
    assert batch.mask[:, 2:4].any(dim=1).all()


def test_make_masked_batch_respects_max_mask_probability():
    clean = torch.arange(24, dtype=torch.long).view(4, 6)
    generator = torch.Generator(device=clean.device).manual_seed(123)
    batch = make_masked_batch(clean, mask_token_id=99, eps=0.1, generator=generator, max_mask_prob=0.4)

    assert batch.mask_prob.max().item() <= 0.4
    assert batch.mask.any(dim=1).all()


def test_make_suffix_eligible_mask_keeps_prefix_fixed():
    clean = torch.arange(24, dtype=torch.long).view(4, 6)
    generator = torch.Generator(device=clean.device).manual_seed(123)
    eligible = make_suffix_eligible_mask(clean, min_prefix_frac=0.5, max_prefix_frac=0.5, generator=generator)

    assert not eligible[:, :3].any()
    assert eligible[:, 3:].all()


def test_make_suffix_span_masks_force_future_without_loss():
    clean = torch.arange(16, dtype=torch.long).view(2, 8)
    generator = torch.Generator(device=clean.device).manual_seed(123)
    eligible, force = make_suffix_span_masks(
        clean,
        span_tokens=3,
        min_prefix_frac=0.25,
        max_prefix_frac=0.25,
        generator=generator,
    )

    assert not eligible[:, :2].any()
    assert eligible[:, 2:5].all()
    assert not eligible[:, 5:].any()
    assert not force[:, :5].any()
    assert force[:, 5:].all()


def test_bidirectional_gpt_diffusion_loss_backward():
    model = build_tiny_bidirectional_model()
    clean = torch.randint(0, 16, (2, 8), dtype=torch.long)
    generator = torch.Generator(device=clean.device).manual_seed(123)

    loss, metrics = masked_diffusion_loss(model, clean, mask_token_id=16, eps=0.1, generator=generator)
    loss.backward()

    assert loss.isfinite()
    assert metrics["mask_fraction"] > 0
    assert model.transformer.wte.weight.grad is not None


def test_diffusion_loss_can_train_suffix_only():
    model = build_tiny_bidirectional_model()
    clean = torch.randint(0, 16, (2, 8), dtype=torch.long)
    generator = torch.Generator(device=clean.device).manual_seed(123)

    loss, metrics = masked_diffusion_loss(
        model,
        clean,
        mask_token_id=16,
        eps=0.9,
        generator=generator,
        mask_pattern="suffix",
        min_prefix_frac=0.5,
        max_prefix_frac=0.5,
    )
    loss.backward()

    assert loss.isfinite()
    assert metrics["mask_fraction"] > 0


def test_diffusion_loss_can_train_suffix_span_only():
    model = build_tiny_bidirectional_model()
    clean = torch.randint(0, 16, (2, 8), dtype=torch.long)
    generator = torch.Generator(device=clean.device).manual_seed(123)

    loss, metrics = masked_diffusion_loss(
        model,
        clean,
        mask_token_id=16,
        eps=0.9,
        generator=generator,
        mask_pattern="suffix_span",
        min_prefix_frac=0.25,
        max_prefix_frac=0.25,
        span_tokens=3,
    )
    loss.backward()

    assert loss.isfinite()
    assert metrics["mask_fraction"] > 0


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


def test_sample_masked_diffusion_repeat_penalty_affects_output():
    model = FixedLogitModel()
    baseline = sample_masked_diffusion(model, mask_token_id=7, length=4, prompt_tokens=[1], steps=3)
    penalized = sample_masked_diffusion(
        model,
        mask_token_id=7,
        length=4,
        prompt_tokens=[1],
        steps=3,
        repeat_penalty=2.0,
    )

    assert baseline == [1, 3, 3, 3]
    assert penalized == [1, 3, 4, 3]


def test_sample_masked_diffusion_no_repeat_ngram_affects_output():
    model = FixedLogitModel()
    baseline = sample_masked_diffusion(model, mask_token_id=7, length=4, prompt_tokens=[1], steps=3)
    no_repeat = sample_masked_diffusion(
        model,
        mask_token_id=7,
        length=4,
        prompt_tokens=[1],
        steps=3,
        no_repeat_ngram_size=2,
    )

    assert baseline == [1, 3, 3, 3]
    assert no_repeat == [1, 3, 3, 4]


def test_sample_masked_diffusion_block_generation_preserves_length():
    model = FixedLogitModel()
    sample = sample_masked_diffusion(
        model,
        mask_token_id=7,
        length=4,
        prompt_tokens=[1],
        steps=3,
        block_size=1,
    )

    assert sample == [1, 3, 3, 3]


def test_sample_masked_diffusion_remasking_preserves_prompt_and_finishes():
    model = build_tiny_bidirectional_model()
    sample = sample_masked_diffusion(
        model,
        mask_token_id=16,
        length=6,
        prompt_tokens=[1, 2],
        steps=4,
        temperature=0.0,
        remask_low_confidence=True,
    )

    assert sample[:2] == [1, 2]
    assert len(sample) == 6
    assert 16 not in sample


def test_sft_loader_wraps_rank_cursor_for_tiny_datasets():
    dataset = TinyConversationDataset([{"messages": []} for _ in range(4)])
    tokenizer = TinyConversationTokenizer()
    args = SimpleNamespace(max_seq_len=5, device_batch_size=1)

    loader = sft_loader(
        dataset,
        tokenizer,
        args,
        device=torch.device("cpu"),
        ddp_rank=7,
        ddp_world_size=8,
    )
    clean_ids, eligible_mask = next(loader)

    assert clean_ids.tolist() == [[1, 2, 3, 0, 0]]
    assert eligible_mask.tolist() == [[False, True, True, False, False]]
