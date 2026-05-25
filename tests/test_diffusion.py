import torch
import torch.nn as nn
from types import SimpleNamespace

from nanochat.diffusion import (
    get_diffusion_vocab_size,
    get_mask_token_id,
    make_masked_batch,
    make_prefix_next_token_masks,
    make_suffix_eligible_mask,
    make_suffix_span_masks,
    masked_diffusion_loss,
    sample_masked_diffusion,
)
from nanochat.gpt import GPT, GPTConfig
from scripts.diffusion_base_train import (
    build_model_meta,
    copy_compatible_initialization_weights,
    make_ar_rollout_sequence_batches,
    make_ar_rollout_span_batch,
)
from scripts.diffusion_chat_sft import sft_loader


class TinyTokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

    def get_vocab_size(self):
        return self.vocab_size


def build_tiny_bidirectional_model(
    vocab_size=17,
    sequence_len=8,
    diffusion_sigma_conditioning=False,
    diffusion_sigma_layer_conditioning=False,
    diffusion_sigma_adaln_conditioning=False,
    diffusion_sigma_embedding="scalar",
    diffusion_sigma_embedding_dim=256,
):
    config = GPTConfig(
        sequence_len=sequence_len,
        vocab_size=vocab_size,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=32,
        window_pattern="L",
        attention_mode="bidirectional",
        diffusion_sigma_conditioning=diffusion_sigma_conditioning,
        diffusion_sigma_layer_conditioning=diffusion_sigma_layer_conditioning,
        diffusion_sigma_adaln_conditioning=diffusion_sigma_adaln_conditioning,
        diffusion_sigma_embedding=diffusion_sigma_embedding,
        diffusion_sigma_embedding_dim=diffusion_sigma_embedding_dim,
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


class FixedTeacherModel(nn.Module):
    def __init__(self, vocab_size=16):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, ids):
        logits = torch.zeros((*ids.shape, self.vocab_size), dtype=torch.float32, device=ids.device)
        logits[..., 7] = 4.0
        logits[..., 3] = 2.0
        return logits


class MaskBiasedLossModel(nn.Module):
    def __init__(self, mask_token_id=4):
        super().__init__()
        self.mask_token_id = mask_token_id
        self.config = GPTConfig(
            sequence_len=4,
            vocab_size=5,
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
        logits = torch.full((*ids.shape, self.config.vocab_size), -5.0, dtype=torch.float32)
        logits[..., 3] = 3.0
        logits[..., self.mask_token_id] = 10.0
        return logits


class PromptCFGModel(nn.Module):
    def __init__(self, mask_token_id=7):
        super().__init__()
        self.mask_token_id = mask_token_id
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
        prompted = ids[:, :1] != self.mask_token_id
        logits[..., 3] = torch.where(prompted, 2.1, 3.0)
        logits[..., 4] = torch.where(prompted, 2.0, 0.0)
        return logits


class PositionRevealModel(nn.Module):
    def __init__(self, mask_token_id=7):
        super().__init__()
        self.mask_token_id = mask_token_id
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
        logits = torch.full((*ids.shape, self.config.vocab_size), -5.0, dtype=torch.float32)
        logits[:, :, 3] = 1.0
        logits[:, :, 4] = 0.5

        logits[:, 1, 3] = 2.0
        logits[:, 1, 4] = 1.9

        previous_revealed = ids[:, 1] != self.mask_token_id
        logits[:, 2, 3] = torch.where(previous_revealed, -5.0, 6.0)
        logits[:, 2, 4] = torch.where(previous_revealed, 6.0, -5.0)
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


def test_make_masked_batch_can_mask_all_eligible_positions():
    clean = torch.arange(12, dtype=torch.long).view(2, 6)
    eligible = torch.zeros_like(clean, dtype=torch.bool)
    eligible[:, 3:] = True
    batch = make_masked_batch(
        clean,
        mask_token_id=99,
        eligible_mask=eligible,
        mask_all_eligible=True,
    )

    assert torch.equal(batch.mask, eligible)
    assert torch.equal(batch.input_ids[eligible], torch.full_like(batch.input_ids[eligible], 99))
    assert torch.equal(batch.targets[eligible], clean[eligible])
    assert torch.equal(batch.targets[~eligible], torch.full_like(batch.targets[~eligible], -1))
    assert torch.equal(batch.mask_prob, torch.ones_like(batch.mask_prob))


def test_make_masked_batch_can_mix_fully_masked_rows():
    clean = torch.arange(12, dtype=torch.long).view(2, 6)
    eligible = torch.zeros_like(clean, dtype=torch.bool)
    eligible[:, 3:] = True
    row_mask_all = torch.tensor([[True], [False]])
    batch = make_masked_batch(
        clean,
        mask_token_id=99,
        eps=0.9,
        generator=torch.Generator(device=clean.device).manual_seed(123),
        eligible_mask=eligible,
        mask_all_eligible=row_mask_all,
    )

    assert torch.equal(batch.mask[0], eligible[0])
    assert torch.equal(batch.mask_prob[0], torch.ones_like(batch.mask_prob[0]))
    assert batch.mask[1, 3:].any()
    assert batch.mask_prob[1] < 1.0


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


def test_make_masked_batch_antithetic_mask_sampling_spreads_probabilities():
    clean = torch.arange(24, dtype=torch.long).view(4, 6)
    generator = torch.Generator(device=clean.device).manual_seed(123)
    batch = make_masked_batch(
        clean,
        mask_token_id=99,
        eps=0.01,
        max_mask_prob=0.81,
        generator=generator,
        mask_sampling="antithetic",
    )

    sorted_probs = batch.mask_prob.flatten().sort().values
    assert torch.allclose(sorted_probs[1:] - sorted_probs[:-1], torch.full((3,), 0.2))
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


def test_make_prefix_next_token_masks_force_future_without_loss():
    clean = torch.arange(16, dtype=torch.long).view(2, 8)
    generator = torch.Generator(device=clean.device).manual_seed(123)
    eligible, force = make_prefix_next_token_masks(
        clean,
        min_prefix_frac=0.25,
        max_prefix_frac=0.25,
        generator=generator,
    )

    assert not eligible[:, :2].any()
    assert eligible[:, 2].all()
    assert not eligible[:, 3:].any()
    assert not force[:, :3].any()
    assert force[:, 3:].all()


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


def test_diffusion_loss_can_train_fully_masked_suffix():
    model = build_tiny_bidirectional_model()
    clean = torch.randint(0, 16, (2, 8), dtype=torch.long)
    generator = torch.Generator(device=clean.device).manual_seed(123)

    loss, metrics = masked_diffusion_loss(
        model,
        clean,
        mask_token_id=16,
        generator=generator,
        mask_pattern="suffix_all",
        min_prefix_frac=0.5,
        max_prefix_frac=0.5,
        loss_normalization="eligible",
    )
    loss.backward()

    assert loss.isfinite()
    assert metrics["mask_fraction"] == 0.5
    assert metrics["mask_prob"] == 1.0
    assert metrics["eligible_fraction"] == 0.5


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


def test_diffusion_loss_can_train_fully_masked_suffix_span():
    model = build_tiny_bidirectional_model()
    clean = torch.randint(0, 16, (2, 8), dtype=torch.long)
    generator = torch.Generator(device=clean.device).manual_seed(123)

    loss, metrics = masked_diffusion_loss(
        model,
        clean,
        mask_token_id=16,
        generator=generator,
        mask_pattern="suffix_span_all",
        min_prefix_frac=0.25,
        max_prefix_frac=0.25,
        span_tokens=3,
        loss_normalization="eligible",
    )
    loss.backward()

    assert loss.isfinite()
    assert metrics["mask_fraction"] == 3 / 8
    assert metrics["mask_prob"] == 1.0
    assert metrics["eligible_fraction"] == 3 / 8


def test_diffusion_loss_can_train_mixed_suffix_span():
    model = build_tiny_bidirectional_model()
    clean = torch.randint(0, 16, (4, 8), dtype=torch.long)
    generator = torch.Generator(device=clean.device).manual_seed(123)

    loss, metrics = masked_diffusion_loss(
        model,
        clean,
        mask_token_id=16,
        generator=generator,
        mask_pattern="suffix_span_mixed",
        min_prefix_frac=0.25,
        max_prefix_frac=0.25,
        span_tokens=3,
        loss_normalization="eligible",
    )
    loss.backward()

    assert loss.isfinite()
    assert metrics["mask_fraction"] > 0
    assert metrics["eligible_fraction"] == 3 / 8


def test_diffusion_loss_can_train_prefix_next_token():
    model = build_tiny_bidirectional_model()
    clean = torch.randint(0, 16, (2, 8), dtype=torch.long)
    generator = torch.Generator(device=clean.device).manual_seed(123)

    loss, metrics = masked_diffusion_loss(
        model,
        clean,
        mask_token_id=16,
        generator=generator,
        mask_pattern="prefix_next",
        min_prefix_frac=0.25,
        max_prefix_frac=0.25,
        loss_normalization="eligible",
    )
    loss.backward()

    assert loss.isfinite()
    assert metrics["mask_fraction"] == 1 / 8
    assert metrics["mask_prob"] == 1.0
    assert metrics["eligible_fraction"] == 1 / 8


def test_diffusion_loss_can_distill_prefix_next_from_teacher():
    model = build_tiny_bidirectional_model()
    teacher = FixedTeacherModel(vocab_size=16)
    clean = torch.randint(0, 16, (2, 8), dtype=torch.long)
    generator = torch.Generator(device=clean.device).manual_seed(123)

    loss, metrics = masked_diffusion_loss(
        model,
        clean,
        mask_token_id=16,
        generator=generator,
        mask_pattern="prefix_next",
        min_prefix_frac=0.25,
        max_prefix_frac=0.25,
        loss_normalization="eligible",
        teacher_model=teacher,
        teacher_kl_weight=0.5,
        teacher_temperature=1.0,
    )
    loss.backward()

    assert loss.isfinite()
    assert metrics["teacher_kl"] > 0
    assert metrics["loss"] > metrics["ce_loss"]
    assert model.transformer.wte.weight.grad is not None


def test_diffusion_loss_can_distill_suffix_span_block_from_teacher():
    model = build_tiny_bidirectional_model()
    teacher = FixedTeacherModel(vocab_size=16)
    clean = torch.randint(0, 16, (2, 8), dtype=torch.long)
    generator = torch.Generator(device=clean.device).manual_seed(123)

    loss, metrics = masked_diffusion_loss(
        model,
        clean,
        mask_token_id=16,
        generator=generator,
        mask_pattern="suffix_span_all",
        min_prefix_frac=0.25,
        max_prefix_frac=0.25,
        span_tokens=3,
        loss_normalization="eligible",
        teacher_model=teacher,
        teacher_kl_weight=0.5,
        teacher_temperature=1.0,
    )
    loss.backward()

    assert loss.isfinite()
    assert metrics["mask_fraction"] == 3 / 8
    assert metrics["teacher_kl"] > 0
    assert metrics["loss"] > metrics["ce_loss"]
    assert model.transformer.wte.weight.grad is not None


def test_diffusion_loss_can_train_explicit_mask_with_teacher_kl_only():
    model = build_tiny_bidirectional_model()
    teacher = FixedTeacherModel(vocab_size=16)
    clean = torch.randint(0, 16, (2, 8), dtype=torch.long)
    eligible_mask = torch.zeros_like(clean, dtype=torch.bool)
    eligible_mask[:, 2:6] = True
    force_mask = torch.zeros_like(clean, dtype=torch.bool)
    force_mask[:, 6:] = True

    loss, metrics = masked_diffusion_loss(
        model,
        clean,
        mask_token_id=16,
        mask_pattern="full",
        eligible_mask=eligible_mask,
        force_mask=force_mask,
        mask_all_eligible=True,
        loss_normalization="eligible",
        teacher_model=teacher,
        teacher_kl_weight=1.0,
        teacher_temperature=1.0,
        ce_loss_weight=0.0,
    )
    loss.backward()

    assert loss.isfinite()
    assert metrics["mask_fraction"] == 4 / 8
    assert metrics["teacher_kl"] > 0
    assert torch.allclose(metrics["loss"], metrics["teacher_kl"])
    assert metrics["ce_loss"] > 0
    assert model.transformer.wte.weight.grad is not None


def test_ar_rollout_span_batch_builds_explicit_training_masks():
    teacher = FixedTeacherModel(vocab_size=16)
    clean = torch.arange(16, dtype=torch.long).view(2, 8)
    args = SimpleNamespace(
        ar_rollout_tokens=3,
        ar_rollout_objective="span",
        ar_rollout_train_tokens=1,
        ar_teacher_model_tag="teacher",
        mask_pattern="suffix_span_all",
        prefix_min_frac=0.25,
        prefix_max_frac=0.25,
        ar_rollout_temperature=0.0,
        ar_rollout_top_k=0,
    )

    rollout_ids, eligible_mask, force_mask = make_ar_rollout_span_batch(teacher, clean, args)

    assert eligible_mask.tolist() == [
        [False, False, True, True, True, False, False, False],
        [False, False, True, True, True, False, False, False],
    ]
    assert force_mask.tolist() == [
        [False, False, False, False, False, True, True, True],
        [False, False, False, False, False, True, True, True],
    ]
    assert rollout_ids[:, 2:5].tolist() == [[7, 7, 7], [7, 7, 7]]


def test_ar_rollout_progressive_batch_keeps_previous_rollout_tokens_visible():
    teacher = FixedTeacherModel(vocab_size=16)
    clean = torch.arange(16, dtype=torch.long).view(2, 8)
    args = SimpleNamespace(
        ar_rollout_tokens=3,
        ar_rollout_objective="progressive",
        ar_rollout_train_tokens=1,
        ar_teacher_model_tag="teacher",
        mask_pattern="suffix_span_all",
        prefix_min_frac=0.25,
        prefix_max_frac=0.25,
        ar_rollout_temperature=0.0,
        ar_rollout_top_k=0,
    )
    generator = torch.Generator(device=clean.device).manual_seed(0)

    rollout_ids, eligible_mask, force_mask = make_ar_rollout_span_batch(teacher, clean, args, generator=generator)

    target_counts = eligible_mask.sum(dim=1)
    assert target_counts.tolist() == [1, 1]
    for row in range(clean.size(0)):
        target_pos = eligible_mask[row].nonzero(as_tuple=False).item()
        assert 2 <= target_pos < 5
        assert not force_mask[row, : target_pos + 1].any()
        assert force_mask[row, target_pos + 1 :].all()
    assert rollout_ids[:, 2:5].tolist() == [[7, 7, 7], [7, 7, 7]]


def test_ar_rollout_sequence_batches_reveal_previous_rollout_tokens():
    teacher = FixedTeacherModel(vocab_size=16)
    clean = torch.arange(16, dtype=torch.long).view(2, 8)
    args = SimpleNamespace(
        ar_rollout_tokens=3,
        ar_rollout_objective="progressive_sequence",
        ar_rollout_train_tokens=2,
        ar_teacher_model_tag="teacher",
        mask_pattern="suffix_span_all",
        prefix_min_frac=0.25,
        prefix_max_frac=0.25,
        ar_rollout_temperature=0.0,
        ar_rollout_top_k=0,
    )
    generator = torch.Generator(device=clean.device).manual_seed(0)

    rollout_ids, mask_pairs = make_ar_rollout_sequence_batches(teacher, clean, args, generator=generator)

    assert len(mask_pairs) == 2
    first_eligible, first_force = mask_pairs[0]
    second_eligible, second_force = mask_pairs[1]
    assert first_eligible.tolist() == [
        [False, False, True, False, False, False, False, False],
        [False, False, True, False, False, False, False, False],
    ]
    assert second_eligible.tolist() == [
        [False, False, False, True, False, False, False, False],
        [False, False, False, True, False, False, False, False],
    ]
    assert first_force.tolist() == [
        [False, False, False, True, True, True, True, True],
        [False, False, False, True, True, True, True, True],
    ]
    assert second_force.tolist() == [
        [False, False, False, False, True, True, True, True],
        [False, False, False, False, True, True, True, True],
    ]
    assert rollout_ids[:, 2:5].tolist() == [[7, 7, 7], [7, 7, 7]]


def test_diffusion_loss_accepts_explicit_force_mask_and_mask_all():
    model = build_tiny_bidirectional_model()
    clean = torch.randint(0, 16, (2, 8), dtype=torch.long)
    eligible_mask = torch.zeros_like(clean, dtype=torch.bool)
    eligible_mask[:, 2:5] = True
    force_mask = torch.zeros_like(clean, dtype=torch.bool)
    force_mask[:, 5:] = True

    loss, metrics = masked_diffusion_loss(
        model,
        clean,
        mask_token_id=16,
        mask_pattern="full",
        eligible_mask=eligible_mask,
        force_mask=force_mask,
        mask_all_eligible=True,
        loss_normalization="eligible",
    )
    loss.backward()

    assert loss.isfinite()
    assert metrics["mask_fraction"] == 3 / 8
    assert metrics["mask_prob"] == 1.0
    assert model.transformer.wte.weight.grad is not None


def test_diffusion_loss_can_train_score_entropy_objective():
    model = build_tiny_bidirectional_model()
    clean = torch.randint(0, 16, (2, 8), dtype=torch.long)
    generator = torch.Generator(device=clean.device).manual_seed(123)

    loss, metrics = masked_diffusion_loss(
        model,
        clean,
        mask_token_id=16,
        eps=0.1,
        max_mask_prob=0.9,
        generator=generator,
        loss_objective="score_entropy",
    )
    loss.backward()

    assert loss.isfinite()
    assert metrics["mask_fraction"] > 0
    assert model.transformer.wte.weight.grad is not None


def test_diffusion_loss_can_train_sigma_scaled_score_entropy_objective():
    model = build_tiny_bidirectional_model()
    clean = torch.randint(0, 16, (2, 8), dtype=torch.long)
    generator = torch.Generator(device=clean.device).manual_seed(123)

    loss, metrics = masked_diffusion_loss(
        model,
        clean,
        mask_token_id=16,
        eps=0.1,
        max_mask_prob=0.9,
        generator=generator,
        loss_objective="score_entropy",
        score_parameterization="sigma_scaled",
    )
    loss.backward()

    assert loss.isfinite()
    assert metrics["mask_fraction"] > 0
    assert model.transformer.wte.weight.grad is not None


def test_diffusion_loss_can_train_sigma_conditioned_score_entropy_objective():
    model = build_tiny_bidirectional_model(diffusion_sigma_conditioning=True)
    clean = torch.randint(0, 16, (2, 8), dtype=torch.long)
    generator = torch.Generator(device=clean.device).manual_seed(123)

    loss, metrics = masked_diffusion_loss(
        model,
        clean,
        mask_token_id=16,
        eps=0.1,
        max_mask_prob=0.9,
        generator=generator,
        loss_objective="score_entropy",
        score_parameterization="sigma_scaled",
    )
    loss.backward()

    assert loss.isfinite()
    assert metrics["mask_fraction"] > 0
    assert model.diffusion_sigma_proj.weight.grad is not None


def test_sigma_conditioned_sampler_passes_current_noise_level():
    model = build_tiny_bidirectional_model(diffusion_sigma_conditioning=True)

    tokens = sample_masked_diffusion(
        model,
        mask_token_id=16,
        length=4,
        prompt_tokens=[1],
        steps=3,
        temperature=0.0,
        seed=123,
    )

    assert len(tokens) == 4
    assert tokens[0] == 1
    assert all(0 <= token < model.config.vocab_size for token in tokens)


def test_diffusion_loss_can_train_layer_sigma_conditioned_score_entropy_objective():
    model = build_tiny_bidirectional_model(diffusion_sigma_layer_conditioning=True)
    clean = torch.randint(0, 16, (2, 8), dtype=torch.long)
    generator = torch.Generator(device=clean.device).manual_seed(123)

    loss, metrics = masked_diffusion_loss(
        model,
        clean,
        mask_token_id=16,
        eps=0.1,
        max_mask_prob=0.9,
        generator=generator,
        loss_objective="score_entropy",
        score_parameterization="sigma_scaled",
    )
    loss.backward()

    assert loss.isfinite()
    assert metrics["mask_fraction"] > 0
    assert model.diffusion_sigma_layer_projs[0].weight.grad is not None


def test_diffusion_loss_can_train_adaln_sigma_conditioned_score_entropy_objective():
    model = build_tiny_bidirectional_model(diffusion_sigma_adaln_conditioning=True)
    clean = torch.randint(0, 16, (2, 8), dtype=torch.long)
    generator = torch.Generator(device=clean.device).manual_seed(123)

    loss, metrics = masked_diffusion_loss(
        model,
        clean,
        mask_token_id=16,
        eps=0.1,
        max_mask_prob=0.9,
        generator=generator,
        loss_objective="score_entropy",
        score_parameterization="sigma_scaled",
    )
    loss.backward()

    assert loss.isfinite()
    assert metrics["mask_fraction"] > 0
    assert model.diffusion_sigma_adaln_projs[0].weight.grad is not None


def test_diffusion_loss_can_train_sinusoidal_adaln_sigma_conditioning():
    model = build_tiny_bidirectional_model(
        diffusion_sigma_adaln_conditioning=True,
        diffusion_sigma_embedding="sinusoidal",
        diffusion_sigma_embedding_dim=16,
    )
    clean = torch.randint(0, 16, (2, 8), dtype=torch.long)
    generator = torch.Generator(device=clean.device).manual_seed(123)

    loss, metrics = masked_diffusion_loss(
        model,
        clean,
        mask_token_id=16,
        eps=0.1,
        max_mask_prob=0.9,
        generator=generator,
        loss_objective="score_entropy",
        score_parameterization="sigma_scaled",
    )
    loss.backward()

    assert loss.isfinite()
    assert metrics["mask_fraction"] > 0
    assert model.diffusion_sigma_embedder.mlp[0].weight.grad is not None
    assert model.diffusion_sigma_adaln_projs[0].weight.grad is not None


def test_diffusion_model_can_initialize_from_causal_vocab_weights():
    source_config = GPTConfig(
        sequence_len=8,
        vocab_size=16,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=32,
        window_pattern="L",
        attention_mode="causal",
    )
    target_config = GPTConfig(
        sequence_len=8,
        vocab_size=17,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=32,
        window_pattern="L",
        attention_mode="bidirectional",
        diffusion_sigma_conditioning=True,
    )
    source = GPT(source_config, pad_vocab_size_to=1)
    target = GPT(target_config, pad_vocab_size_to=1)
    source.init_weights()
    target.init_weights()
    original_mask_embedding = target.transformer.wte.weight[16].detach().clone()
    original_mask_head = target.lm_head.weight[16].detach().clone()

    copied, skipped = copy_compatible_initialization_weights(target, source.state_dict())

    assert copied
    assert any(name.startswith("diffusion_sigma_proj") for name, _reason in skipped)
    assert torch.equal(target.transformer.wte.weight[:16], source.transformer.wte.weight)
    assert torch.equal(target.lm_head.weight[:16], source.lm_head.weight)
    assert torch.equal(target.transformer.wte.weight[16], original_mask_embedding)
    assert torch.equal(target.lm_head.weight[16], original_mask_head)


def test_diffusion_loss_can_normalize_by_eligible_tokens():
    model = build_tiny_bidirectional_model()
    clean = torch.randint(0, 16, (2, 8), dtype=torch.long)

    all_loss, all_metrics = masked_diffusion_loss(
        model,
        clean,
        mask_token_id=16,
        eps=0.9,
        generator=torch.Generator(device=clean.device).manual_seed(123),
        mask_pattern="suffix_span",
        min_prefix_frac=0.25,
        max_prefix_frac=0.25,
        span_tokens=2,
        loss_normalization="all",
    )
    eligible_loss, eligible_metrics = masked_diffusion_loss(
        model,
        clean,
        mask_token_id=16,
        eps=0.9,
        generator=torch.Generator(device=clean.device).manual_seed(123),
        mask_pattern="suffix_span",
        min_prefix_frac=0.25,
        max_prefix_frac=0.25,
        span_tokens=2,
        loss_normalization="eligible",
    )

    assert eligible_metrics["eligible_fraction"] == 0.25
    assert all_metrics["eligible_fraction"] == 0.25
    assert torch.allclose(eligible_loss, all_loss * 4)


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


def test_diffusion_loss_excludes_mask_token_from_targets():
    model = MaskBiasedLossModel(mask_token_id=4)
    clean = torch.full((2, 4), 3, dtype=torch.long)
    eligible = torch.ones_like(clean, dtype=torch.bool)

    loss, _metrics = masked_diffusion_loss(
        model,
        clean,
        mask_token_id=4,
        eligible_mask=eligible,
        generator=torch.Generator(device=clean.device).manual_seed(123),
    )

    assert loss.item() < 0.01


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
    assert penalized[:1] == [1]
    assert penalized != baseline
    assert 4 in penalized[1:]


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


def test_sample_masked_diffusion_cfg_scale_affects_prompt_conditioning():
    model = PromptCFGModel(mask_token_id=7)
    baseline = sample_masked_diffusion(model, mask_token_id=7, length=4, prompt_tokens=[1], steps=3)
    guided = sample_masked_diffusion(model, mask_token_id=7, length=4, prompt_tokens=[1], steps=3, cfg_scale=1.0)

    assert baseline == [1, 3, 3, 3]
    assert guided == [1, 4, 4, 4]


def test_sample_masked_diffusion_reveal_strategy_affects_schedule():
    model = PositionRevealModel(mask_token_id=7)
    confidence = sample_masked_diffusion(model, mask_token_id=7, length=4, prompt_tokens=[1], steps=3)
    left_to_right = sample_masked_diffusion(
        model,
        mask_token_id=7,
        length=4,
        prompt_tokens=[1],
        steps=3,
        reveal_strategy="left_to_right",
    )

    assert confidence == [1, 3, 3, 3]
    assert left_to_right == [1, 3, 4, 3]


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


def test_sample_masked_diffusion_sedd_analytic_preserves_prompt_and_finishes():
    model = FixedLogitModel()
    sample = sample_masked_diffusion(
        model,
        mask_token_id=7,
        length=4,
        prompt_tokens=[1],
        steps=3,
        sampler="sedd_analytic",
        seed=123,
    )

    assert len(sample) == 4
    assert sample[0] == 1
    assert 7 not in sample[1:]


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


def test_sample_masked_diffusion_random_remasking_preserves_prompt_and_finishes():
    model = build_tiny_bidirectional_model()
    sample = sample_masked_diffusion(
        model,
        mask_token_id=16,
        length=6,
        prompt_tokens=[1, 2],
        steps=4,
        temperature=0.0,
        remask_strategy="random",
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


def test_diffusion_train_builds_causal_model_when_requested():
    args = SimpleNamespace(
        depth=2,
        aspect_ratio=16,
        head_dim=16,
        max_seq_len=8,
        attention_mode="causal",
        diffusion_sigma_conditioning=False,
        diffusion_sigma_layer_conditioning=False,
        diffusion_sigma_adaln_conditioning=False,
        diffusion_sigma_embedding="scalar",
        diffusion_sigma_embedding_dim=256,
    )

    model = build_model_meta(args, vocab_size=17)

    assert model.config.attention_mode == "causal"
    assert all(window[1] == 0 for window in model.window_sizes)
