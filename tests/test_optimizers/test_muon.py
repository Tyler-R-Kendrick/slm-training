"""SLM-222: Muon/AdamW hybrid optimizer regression tests."""

from __future__ import annotations

from pathlib import Path

import hashlib
import json

import pytest

torch = pytest.importorskip("torch")

from slm_training.dsl.schema import ExampleRecord, write_jsonl
from slm_training.harnesses.model_build import train
from slm_training.harnesses.model_build.config import ModelBuildConfig
from slm_training.harnesses.model_build.full_state import load_full_state
from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel
from slm_training.optimizers.muon import (
    build_muon_hybrid,
    newton_schulz_orthogonalize,
)

HERO = 'root = Stack([hero], "column")\nhero_title = TextContent(":hero.title")\nhero_body = TextContent(":hero.body")\nhero = Card([hero_title, hero_body])'


def _write_train_dir(path: Path, records: list[ExampleRecord]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    records_path = path / "records.jsonl"
    write_jsonl(records_path, records)
    content = records_path.read_bytes()
    manifest = {
        "version": "test-fixture",
        "kind": "train",
        "records": str(records_path),
        "record_count": len(records),
        "content_fingerprint": hashlib.sha256(content).hexdigest(),
    }
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_newton_schulz_orthogonalizes_columns_for_tall_matrix() -> None:
    G = torch.randn(16, 8)
    orth = newton_schulz_orthogonalize(G, steps=10)
    identity = orth.T @ orth
    torch.testing.assert_close(identity, torch.eye(8), atol=1e-3, rtol=1e-3)


def test_newton_schulz_orthogonalizes_rows_for_wide_matrix() -> None:
    G = torch.randn(4, 16)
    orth = newton_schulz_orthogonalize(G, steps=10)
    identity = orth @ orth.T
    torch.testing.assert_close(identity, torch.eye(4), atol=1e-3, rtol=1e-3)


def test_muon_partition_respects_default_deny_list() -> None:
    model = TwoTowerModel.from_records(
        [ExampleRecord(id="a", prompt="Hero", openui=HERO, split="train")],
        config=TwoTowerConfig(
            d_model=32,
            n_heads=4,
            context_layers=1,
            denoiser_layers=1,
            context_backend="scratch",
            denoiser_backend="scratch",
            grammar_constrained=False,
            seed=0,
        ),
        device="cpu",
    )
    opt = build_muon_hybrid(model.named_parameters(), lr=1e-3)
    muon_names = {id(p) for g in opt.param_groups if g.get("optimizer") == "muon" for p in g["params"]}
    adamw_names = {id(p) for g in opt.param_groups if g.get("optimizer") == "adamw" for p in g["params"]}
    assert muon_names and adamw_names
    # Embeddings and output heads must stay on AdamW.
    assert id(model.denoiser.tok.weight) in adamw_names
    assert id(model.denoiser.lm_head.weight) in adamw_names
    # Dense 2-D matrices should be on Muon.
    for name, p in model.named_parameters():
        if (
            p.ndim == 2
            and "tok" not in name
            and "lm_head" not in name
            and "pos" not in name
            and "norm" not in name
            and "bias" not in name
            and not name.startswith(
                (
                    "length_head",
                    "component_inventory_head",
                    "component_plan_head",
                    "slot_component_head",
                    "component_edge_head",
                    "binder_",
                    "root_reference_",
                    "trust_gate",
                    "survival_head",
                )
            )
        ):
            assert id(p) in muon_names, f"{name} should be Muon-eligible"


def test_muon_step_updates_weights_without_nan() -> None:
    model = TwoTowerModel.from_records(
        [ExampleRecord(id="a", prompt="Hero", openui=HERO, split="train")],
        config=TwoTowerConfig(
            d_model=16,
            n_heads=2,
            context_layers=1,
            denoiser_layers=1,
            context_backend="scratch",
            denoiser_backend="scratch",
            grammar_constrained=False,
            seed=0,
        ),
        device="cpu",
    )
    opt = build_muon_hybrid(model.named_parameters(), lr=1e-3)
    before = {id(p): p.detach().clone() for p in model.parameters() if p.requires_grad}
    loss = model.training_loss([ExampleRecord(id="a", prompt="Hero", openui=HERO, split="train")])
    loss.backward()
    opt.step()
    changed = 0
    for p in model.parameters():
        if p.requires_grad and id(p) in before:
            assert torch.isfinite(p).all()
            if not torch.equal(p, before[id(p)]):
                changed += 1
    assert changed > 0


def test_build_muon_hybrid_flat_or_grouped_parameters() -> None:
    model = TwoTowerModel.from_records(
        [ExampleRecord(id="a", prompt="Hero", openui=HERO, split="train")],
        config=TwoTowerConfig(d_model=16, n_heads=2, context_layers=1, denoiser_layers=1),
        device="cpu",
    )
    named = list(model.named_parameters())
    opt = build_muon_hybrid(named, lr=1e-3)
    covered = {id(p) for g in opt.param_groups for p in g["params"]}
    expected = {id(p) for _, p in named if p.requires_grad}
    assert covered == expected


def test_adamw_default_parity() -> None:
    config = ModelBuildConfig(
        train_dir=Path("."),
        optimizer_name="adamw",
        lr=1e-3,
        weight_decay=0.01,
    )
    # The factory path is exercised in train_loop; here we just verify config plumbing.
    assert config.optimizer_name == "adamw"
    assert config.weight_decay == 0.01


def test_muon_full_state_resume_roundtrip(tmp_path: Path) -> None:
    config = ModelBuildConfig(
        train_dir=tmp_path / "train",
        test_dir=None,
        suite="smoke",
        run_root=tmp_path / "runs",
        run_id="muon-resume",
        steps=1,
        batch_size=1,
        lr=1e-3,
        optimizer_name="muon_hybrid",
        muon_lr=5e-4,
        adamw_lr=1e-3,
        weight_decay=0.0,
        device="cpu",
        model_name="twotower",
        d_model=16,
        n_heads=2,
        context_layers=1,
        denoiser_layers=1,
        context_backend="scratch",
        denoiser_backend="scratch",
        grammar_constrained=False,
        full_state_checkpoint=True,
        seed=0,
    )
    _write_train_dir(
        config.train_dir,
        [ExampleRecord(id="a", prompt="Hero", openui=HERO, split="train")],
    )

    result = train(config)
    run_dir = config.run_root / config.run_id
    ckpt = run_dir / "checkpoints" / "last_full_state.pt"
    assert ckpt.exists()

    payload = load_full_state(ckpt)
    assert payload.get("optimizer_fingerprint") is not None
    assert payload["optimizer_fingerprint"]["optimizer"] == "muon_hybrid"

    # Resume with same config.
    config.resume_from = ckpt
    result2 = train(config)
    assert result2["steps"] >= result["steps"]


def test_cross_optimizer_resume_is_fail_closed(tmp_path: Path) -> None:
    config = ModelBuildConfig(
        train_dir=tmp_path / "train",
        test_dir=None,
        suite="smoke",
        run_root=tmp_path / "runs",
        run_id="muon-cross",
        steps=1,
        batch_size=1,
        lr=1e-3,
        optimizer_name="adamw",
        device="cpu",
        model_name="twotower",
        d_model=16,
        n_heads=2,
        context_layers=1,
        denoiser_layers=1,
        context_backend="scratch",
        denoiser_backend="scratch",
        grammar_constrained=False,
        full_state_checkpoint=True,
        seed=0,
    )
    _write_train_dir(
        config.train_dir,
        [ExampleRecord(id="a", prompt="Hero", openui=HERO, split="train")],
    )

    _ = train(config)
    run_dir = config.run_root / config.run_id
    ckpt = run_dir / "checkpoints" / "last_full_state.pt"
    assert ckpt.exists()

    config.optimizer_name = "muon_hybrid"
    config.resume_from = ckpt
    with pytest.raises(ValueError, match="optimizer"):
        train(config)
