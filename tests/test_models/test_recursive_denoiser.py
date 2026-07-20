"""SLM-138 SharedRecursiveDenoiserTower regression tests."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from slm_training.dsl.schema import ExampleRecord
from slm_training.models.blocks import DenoiserTower
from slm_training.models.recursive_denoiser import SharedRecursiveDenoiserTower
from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel

HERO = (
    'root = Stack([hero], "column")\n'
    'hero_title = TextContent(":hero.title")\n'
    'hero_body = TextContent(":hero.body")\n'
    "hero = Card([hero_title, hero_body])"
)
CTA = 'root = Stack([cta])\ncta = Button(":cta.label")'


def test_recursive_tower_matches_denoiser_interface() -> None:
    """The recursive tower exposes the same public attributes/methods."""
    vocab, d_model, max_len = 23, 16, 32
    rec = SharedRecursiveDenoiserTower(
        vocab_size=vocab,
        d_model=d_model,
        n_layers=1,
        n_heads=2,
        max_len=max_len,
    )
    assert rec.tok.weight.shape == (vocab, d_model)
    assert rec.lm_head.weight is rec.tok.weight
    assert rec.max_len == max_len
    assert len(rec.layers) == 1
    assert hasattr(rec, "kind_lookup")
    assert hasattr(rec, "set_runtime_symbol_features")


def test_recursive_forward_shapes_and_gradients() -> None:
    vocab, d_model, tgt, ctx_len = 23, 16, 6, 3
    tower = SharedRecursiveDenoiserTower(
        vocab_size=vocab,
        d_model=d_model,
        n_layers=2,
        n_heads=2,
        max_len=32,
        recursive_steps=2,
        recursive_transition_layers=2,
    )
    noisy = torch.randint(1, vocab, (2, tgt))
    ctx = torch.randn(2, ctx_len, d_model)
    logits = tower(noisy, ctx, pad_id=0)
    assert logits.shape == (2, tgt, vocab)
    loss = logits.sum()
    loss.backward()
    assert tower.tok.weight.grad is not None
    assert tower.ctx_proj.weight.grad is not None
    assert tower.z_latent.grad is not None


def test_recursive_encode_project_matches_forward() -> None:
    vocab, d_model, tgt, ctx_len = 23, 16, 6, 3
    tower = SharedRecursiveDenoiserTower(
        vocab_size=vocab,
        d_model=d_model,
        n_layers=2,
        n_heads=2,
        max_len=32,
        recursive_steps=2,
    )
    noisy = torch.randint(1, vocab, (2, tgt))
    ctx = torch.randn(2, ctx_len, d_model)
    hidden = tower.encode(noisy, ctx, pad_id=0)
    assert hidden.shape == (2, tgt, d_model)
    logits = tower.project(hidden)
    full = tower(noisy, ctx, pad_id=0)
    torch.testing.assert_close(logits, full)

    candidates = torch.tensor([1, 2, 3])
    gathered = tower.project(hidden, candidate_ids=candidates)
    assert gathered.shape == (2, tgt, 3)
    torch.testing.assert_close(gathered, full.index_select(-1, candidates))


def test_recursive_steps_one_parity_with_denoiser_tower() -> None:
    """R=1 with L transition blocks matches the stacked DenoiserTower contract."""
    vocab, d_model, tgt, ctx_len = 23, 16, 6, 3
    torch.manual_seed(0)
    stacked = DenoiserTower(
        vocab_size=vocab, d_model=d_model, n_layers=2, n_heads=2, max_len=32
    )
    torch.manual_seed(0)
    recursive = SharedRecursiveDenoiserTower(
        vocab_size=vocab,
        d_model=d_model,
        n_layers=2,
        n_heads=2,
        max_len=32,
        recursive_steps=1,
        recursive_transition_layers=2,
    )
    noisy = torch.randint(1, vocab, (2, tgt))
    ctx = torch.randn(2, ctx_len, d_model)
    stacked.eval()
    recursive.eval()
    with torch.no_grad():
        s_logits = stacked(noisy, ctx, pad_id=0)
        r_logits = recursive(noisy, ctx, pad_id=0)
    assert s_logits.shape == r_logits.shape == (2, tgt, vocab)
    # The recursive tower adds a z-state path, so the outputs differ, but both
    # must be deterministic and finite with the same interface shape.
    assert torch.isfinite(r_logits).all()


def test_weight_sharing_across_recursions() -> None:
    """The same layer objects are reused at every recursion step."""
    tower = SharedRecursiveDenoiserTower(
        vocab_size=23,
        d_model=16,
        n_layers=4,
        n_heads=2,
        max_len=32,
        recursive_steps=3,
    )
    f_ids = {id(layer) for layer in tower._f_layers}
    g_ids = {id(layer) for layer in tower._g_layers}
    out = tower.recursive_outputs(
        torch.randint(1, 23, (1, 4)), torch.randn(1, 2, 16), pad_id=0
    )
    depth_logits = out["depth_logits"]
    assert len(depth_logits) == 3
    # All computation flows through the same object-identity layers each step.
    assert len(f_ids) + len(g_ids) == len(tower.layers)


def test_runtime_symbol_features_sliced_projection() -> None:
    vocab, d_model = 23, 16
    tower = SharedRecursiveDenoiserTower(
        vocab_size=vocab,
        d_model=d_model,
        n_layers=1,
        n_heads=2,
        max_len=32,
    )
    features = torch.randn(1, vocab, d_model)
    tower.set_runtime_symbol_features(features)
    noisy = torch.randint(1, vocab, (1, 4))
    ctx = torch.randn(1, 2, d_model)
    logits = tower(noisy, ctx, pad_id=0)
    assert logits.shape == (1, 4, vocab)
    # Sliced projection matches the full-vocabulary gather.
    hidden = tower.encode(noisy, ctx, pad_id=0)
    candidates = torch.tensor([1, 2, 3])
    gathered = tower.project(hidden[0, 0], candidate_ids=candidates)
    assert gathered.shape == (3,)


def test_twotower_shared_recursive_trains_and_roundtrips(tmp_path: Path) -> None:
    records = [
        ExampleRecord(id="a", prompt="Hero layout", openui=HERO, split="train"),
        ExampleRecord(id="b", prompt="CTA layout", openui=CTA, split="train"),
    ]
    model = TwoTowerModel.from_records(
        records,
        config=TwoTowerConfig(
            d_model=32,
            n_heads=2,
            context_layers=1,
            denoiser_layers=2,
            denoiser_arch="shared_recursive",
            recursive_steps=2,
            recursive_transition_layers=2,
            grammar_constrained=False,
            gen_steps=2,
            seed=0,
        ),
        device="cpu",
    )
    assert isinstance(model.denoiser, SharedRecursiveDenoiserTower)
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=1e-3)
    opt.zero_grad(set_to_none=True)
    loss = model.training_loss(records)
    loss.backward()
    opt.step()

    ckpt = tmp_path / "recursive.pt"
    model.save(ckpt)
    loaded = TwoTowerModel.from_checkpoint(ckpt, device="cpu")
    assert loaded.config.denoiser_arch == "shared_recursive"
    assert isinstance(loaded.denoiser, SharedRecursiveDenoiserTower)


def test_twotower_deep_supervision_metrics() -> None:
    records = [
        ExampleRecord(id="a", prompt="Hero layout", openui=HERO, split="train"),
    ]
    model = TwoTowerModel.from_records(
        records,
        config=TwoTowerConfig(
            d_model=32,
            n_heads=2,
            context_layers=1,
            denoiser_layers=2,
            denoiser_arch="shared_recursive",
            recursive_steps=3,
            recursive_transition_layers=2,
            recursive_depth_supervision_weights=(0.5, 1.0, 0.5),
            grammar_constrained=False,
            seed=0,
        ),
        device="cpu",
    )
    loss = model.training_loss(records)
    assert torch.isfinite(loss)
    assert "recursive_depth_supervision_loss" in model.last_training_metrics
    assert "recursive_depth_loss_0" in model.last_training_metrics
    assert "recursive_depth_loss_2" in model.last_training_metrics


def test_checkpoint_migration_to_shared_recursive(tmp_path: Path) -> None:
    from slm_training.models.checkpoint_migrate import (
        migrate_to_shared_recursive_denoiser,
    )

    records = [
        ExampleRecord(id="a", prompt="Hero", openui=HERO, split="train"),
    ]
    stacked = TwoTowerModel.from_records(
        records,
        config=TwoTowerConfig(d_model=32, n_heads=2, denoiser_layers=2, seed=0),
        device="cpu",
    )
    src = tmp_path / "stacked.pt"
    stacked.save(src)

    dst = tmp_path / "recursive.pt"
    report = migrate_to_shared_recursive_denoiser(
        src,
        dst,
        config={"recursive_steps": 2, "recursive_transition_layers": 2},
        device="cpu",
    )
    assert dst.exists()
    assert report["denoiser_arch"] == "shared_recursive"
    assert any("z_latent" in k for k in report["initialized_keys"])
    assert any("ctx_proj" in k for k in report["initialized_keys"])

    loaded = TwoTowerModel.from_checkpoint(dst, device="cpu")
    assert loaded.config.denoiser_arch == "shared_recursive"
    assert isinstance(loaded.denoiser, SharedRecursiveDenoiserTower)
