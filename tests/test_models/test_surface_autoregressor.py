"""Regression tests for VSS3-05 surface autoregressor (SLM-73).

Torch-only. Skipped where torch is unavailable, per this repo's convention
(``pytest.importorskip("torch")``); the assertions are CI-run.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from slm_training.models.surface_autoregressor import (
    DecorativeConstraint,
    IdentifierConstraint,
    SurfaceAutoregressor,
    SurfaceAutoregressorConfig,
    SurfaceByteVocab,
    train_surface_autoregressor,
)


@pytest.fixture
def vocab() -> SurfaceByteVocab:
    return SurfaceByteVocab()


@pytest.fixture
def tiny_config() -> SurfaceAutoregressorConfig:
    return SurfaceAutoregressorConfig(d_model=32, n_layers=1, n_heads=2, max_len=32)


def test_byte_vocab_encode_decode_roundtrip(vocab: SurfaceByteVocab) -> None:
    text = "hello_world"
    ids = vocab.encode(text)
    assert ids[0] == vocab.bos_id
    assert ids[-1] == vocab.eos_id
    assert vocab.decode(ids) == text


def test_byte_vocab_decode_ignores_specials(vocab: SurfaceByteVocab) -> None:
    text = "abc"
    ids = vocab.encode(text)
    assert vocab.decode([vocab.pad_id, *ids, vocab.unk_id]) == text


def test_identifier_constraint_first_position(vocab: SurfaceByteVocab) -> None:
    constraint = IdentifierConstraint(vocab, max_bytes=16)
    allowed = constraint.allowed_next("")
    assert vocab.eos_id not in allowed
    # Digits are not allowed at position 0.
    assert vocab.byte_to_id["0"] not in allowed
    assert vocab.byte_to_id["a"] in allowed
    assert vocab.byte_to_id["_"] in allowed


def test_identifier_constraint_later_positions_allow_eos(vocab: SurfaceByteVocab) -> None:
    constraint = IdentifierConstraint(vocab, max_bytes=16)
    allowed = constraint.allowed_next("h")
    assert vocab.byte_to_id["0"] in allowed
    assert vocab.eos_id in allowed


def test_identifier_constraint_rejects_reserved(vocab: SurfaceByteVocab) -> None:
    constraint = IdentifierConstraint(vocab, max_bytes=16, reserved={"hello"})
    assert not constraint.is_complete("hello")
    allowed = constraint.allowed_next("hello")
    assert vocab.eos_id not in allowed


def test_identifier_constraint_rejects_collision(vocab: SurfaceByteVocab) -> None:
    constraint = IdentifierConstraint(vocab, max_bytes=16, peers={"foo"})
    assert not constraint.is_complete("foo")


def test_decorative_constraint_allows_printable_and_eos(vocab: SurfaceByteVocab) -> None:
    constraint = DecorativeConstraint(vocab, max_bytes=8)
    allowed = constraint.allowed_next("hi")
    assert vocab.byte_to_id["!"] in allowed
    assert vocab.eos_id in allowed


def test_untrained_model_generate_returns_none_or_fallback(tiny_config: SurfaceAutoregressorConfig, vocab: SurfaceByteVocab) -> None:
    model = SurfaceAutoregressor(tiny_config)
    constraint = IdentifierConstraint(vocab, max_bytes=16)
    prompt_ids = torch.tensor(vocab.encode("kind=internal_identifier symbol=title"), dtype=torch.long)
    value = model.generate(prompt_ids, constraint, max_bytes=16)
    # Untrained model is extremely unlikely to produce a legal identifier.
    assert value is None or constraint.is_complete(value)


def test_tiny_fixture_overfit(tiny_config: SurfaceAutoregressorConfig, vocab: SurfaceByteVocab) -> None:
    model = SurfaceAutoregressor(tiny_config)
    # Use a single example so the tiny fixture model reliably overfits.
    examples = [("kind=internal_identifier symbol=title", "title")]
    metrics = train_surface_autoregressor(model, examples, steps=200, lr=5e-3, seed=0)
    assert metrics["final_loss"] < metrics["initial_loss"]

    # Greedy generation should recover the trained target.
    model.eval()
    constraint = IdentifierConstraint(vocab, max_bytes=16)
    prompt_ids = torch.tensor(vocab.encode("kind=internal_identifier symbol=title"), dtype=torch.long)
    value = model.generate(prompt_ids, constraint, max_bytes=16, temperature=0.0, top_k=1)
    assert value == "title"


def test_model_save_load_roundtrip(tmp_path: pytest.TempPathFactory, tiny_config: SurfaceAutoregressorConfig) -> None:
    model = SurfaceAutoregressor(tiny_config)
    path = tmp_path / "surface_ar.pt"  # type: ignore[attr-defined]
    model.save(path)
    loaded = SurfaceAutoregressor.load(path, device="cpu")
    assert loaded.config.d_model == model.config.d_model
    assert sum(p.numel() for p in loaded.parameters()) == sum(
        p.numel() for p in model.parameters()
    )


def test_from_records_ignores_records(tiny_config: SurfaceAutoregressorConfig) -> None:
    model = SurfaceAutoregressor.from_records([], config=tiny_config)
    assert model.config == tiny_config
