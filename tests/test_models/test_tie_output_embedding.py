"""SLM-211: output-head tying control regression tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from slm_training.dsl.schema import ExampleRecord
from slm_training.harnesses.experiments.slm214_spectral_snapshot import (
    run_spectral_snapshot_fixture,
)
from slm_training.models.blocks import DenoiserTower
from slm_training.models.recursive_denoiser import SharedRecursiveDenoiserTower
from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel
from slm_training.runtime.cactus import export_checkpoint_bundle

HERO = 'root = Stack([b3], "column")\nb1 = TextContent(":slot_0")\nb2 = TextContent(":slot_1")\nb3 = Card([b1, b2])'


def _tiny_model(tmp_path: Path, *, tie_output_embedding: bool = True) -> TwoTowerModel:
    records = [ExampleRecord(id="a", prompt="Hero", openui=HERO, split="train")]
    return TwoTowerModel.from_records(
        records,
        config=TwoTowerConfig(
            d_model=32,
            n_heads=4,
            context_layers=1,
            denoiser_layers=1,
            context_backend="scratch",
            denoiser_backend="scratch",
            grammar_constrained=False,
            gen_steps=2,
            seed=0,
            tie_output_embedding=tie_output_embedding,
        ),
        device="cpu",
    )


def test_default_tied_shares_storage() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        model = _tiny_model(Path(tmp), tie_output_embedding=True)
    assert isinstance(model.denoiser, DenoiserTower)
    assert model.config.tie_output_embedding is True
    assert model.denoiser.tie_output_embedding is True
    assert model.denoiser.lm_head.weight is model.denoiser.tok.weight


def test_untied_distinct_storage_and_equal_initial_logits() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        model = _tiny_model(Path(tmp), tie_output_embedding=False)
    assert isinstance(model.denoiser, DenoiserTower)
    assert model.config.tie_output_embedding is False
    assert model.denoiser.tie_output_embedding is False
    assert model.denoiser.lm_head.weight is not model.denoiser.tok.weight
    torch.testing.assert_close(
        model.denoiser.lm_head.weight, model.denoiser.tok.weight
    )


def test_untied_recursive_and_hf_towers_accept_flag() -> None:
    rec = SharedRecursiveDenoiserTower(
        vocab_size=23, d_model=16, n_layers=1, n_heads=2, max_len=32,
        tie_output_embedding=False,
    )
    assert rec.tie_output_embedding is False
    assert rec.lm_head.weight is not rec.tok.weight


def test_optimizer_groups_have_no_duplicate_storage(tmp_path: Path) -> None:
    model = _tiny_model(tmp_path, tie_output_embedding=True)
    groups = model.optimizer_parameter_groups()
    seen: set[int] = set()
    total = 0
    for g in groups:
        for p in g["params"]:
            assert id(p) not in seen
            seen.add(id(p))
            total += p.numel()
    trainable = list(model.trainable_parameters())
    assert sum(p.numel() for p in trainable) == total


def test_save_load_roundtrip_preserves_tie_mode(tmp_path: Path) -> None:
    for tie in (True, False):
        model = _tiny_model(tmp_path, tie_output_embedding=tie)
        ckpt = tmp_path / f"tie-{tie}.pt"
        model.save(ckpt)
        loaded = TwoTowerModel.from_checkpoint(ckpt, device="cpu")
        assert loaded.config.tie_output_embedding is tie
        if tie:
            assert loaded.denoiser.lm_head.weight is loaded.denoiser.tok.weight
        else:
            assert loaded.denoiser.lm_head.weight is not loaded.denoiser.tok.weight


def test_cross_tie_load_is_fail_closed_without_explicit_migration(tmp_path: Path) -> None:
    tied_model = _tiny_model(tmp_path, tie_output_embedding=True)
    untied_model = _tiny_model(tmp_path, tie_output_embedding=False)
    tied_ckpt = tmp_path / "tied.pt"
    untied_ckpt = tmp_path / "untied.pt"
    tied_model.save(tied_ckpt)
    untied_model.save(untied_ckpt)

    # Loading a tied checkpoint into an already-built untied model must fail closed.
    with pytest.raises(ValueError, match="tie_output_embedding mismatch"):
        untied_model.load(tied_ckpt)
    untied_model.load(tied_ckpt, allow_tie_migration=True)
    assert untied_model.config.tie_output_embedding is False
    assert untied_model.denoiser.lm_head.weight is not untied_model.denoiser.tok.weight
    torch.testing.assert_close(
        untied_model.denoiser.lm_head.weight, untied_model.denoiser.tok.weight
    )

    # Loading an untied checkpoint into a tied model must also fail closed.
    with pytest.raises(ValueError, match="tie_output_embedding mismatch"):
        tied_model.load(untied_ckpt)
    tied_model.load(untied_ckpt, allow_tie_migration=True)
    assert tied_model.config.tie_output_embedding is True
    assert tied_model.denoiser.lm_head.weight is tied_model.denoiser.tok.weight


def test_spectral_snapshot_records_tie_output_embedding(tmp_path: Path) -> None:
    model = _tiny_model(tmp_path, tie_output_embedding=False)
    report = run_spectral_snapshot_fixture(
        model, null_draws=5, max_matrices=4, device="cpu"
    )
    assert report.snapshots
    assert all(
        s.tie_output_embedding is False for s in report.snapshots
    )


def test_cactus_manifest_records_tie_output_embedding(tmp_path: Path) -> None:
    model = _tiny_model(tmp_path, tie_output_embedding=False)
    ckpt = tmp_path / "cactus.pt"
    model.save(ckpt)
    bundle_dir = tmp_path / "bundle"
    export_checkpoint_bundle(ckpt, bundle_dir)
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest.get("tie_output_embedding") is False
