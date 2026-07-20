"""Tests for SLM-158 (SPV3-05) sequence-mixer comparison fixture harness."""

from __future__ import annotations

import pytest
import torch

from slm_training.harnesses.experiments.slm158_mixer_comparison import (
    CommonConfig,
    MixerArm,
    MixerFamily,
    MixerManifest,
    MixerReportRow,
    _MixerClassifier,
    _build_mixer,
    _make_dataset,
    build_manifest,
    run_fixture_campaign,
    validate_manifest,
)


@pytest.fixture
def manifest() -> MixerManifest:
    return build_manifest()


def test_build_manifest_has_required_arms(manifest: MixerManifest) -> None:
    arm_ids = {arm.arm_id for arm in manifest.arms}
    assert "T0_no_mixer" in arm_ids
    assert "T1_transformer" in arm_ids
    assert "S1_mamba_reference" in arm_ids
    assert "L1_gated_delta_net" in arm_ids


def test_validate_manifest_passes_for_default(manifest: MixerManifest) -> None:
    assert validate_manifest(manifest) == []


def test_validate_manifest_rejects_duplicate_arm_ids(manifest: MixerManifest) -> None:
    arms = list(manifest.arms)
    duplicated = arms + [arms[0]]
    bad = MixerManifest(arms=tuple(duplicated))
    errors = validate_manifest(bad)
    assert any("duplicate arm_id" in e for e in errors)


def test_validate_manifest_rejects_promotable_blocked(manifest: MixerManifest) -> None:
    arms = list(manifest.arms)
    blocked = MixerArm(
        **{
            **arms[0].__dict__,
            "blocked": True,
            "blocker": "test blocker",
        }
    )
    new_arms = [blocked if arm.arm_id == blocked.arm_id else arm for arm in arms]
    bad = MixerManifest(arms=tuple(new_arms))
    errors = validate_manifest(bad)
    assert any(blocked.arm_id in e for e in errors)


def test_common_config_roundtrip() -> None:
    cfg = CommonConfig()
    data = cfg.to_dict()
    restored = CommonConfig.from_dict(data)
    assert restored.d_model == cfg.d_model
    assert restored.seeds == cfg.seeds


def test_manifest_to_dict_roundtrip(manifest: MixerManifest) -> None:
    data = manifest.to_dict()
    restored = MixerManifest.from_dict(data)
    assert restored.matrix_set == manifest.matrix_set
    assert {a.arm_id for a in restored.arms} == {a.arm_id for a in manifest.arms}


def test_mixer_encode_contract(manifest: MixerManifest) -> None:
    pytest.importorskip("torch")
    cfg = CommonConfig(d_model=16, seq_len=8)
    for arm in manifest.arms:
        if arm.blocked:
            continue
        mixer = _build_mixer(arm.family, cfg)
        x = torch.randn(2, cfg.seq_len, cfg.d_model)
        pad_mask = torch.zeros(2, cfg.seq_len, dtype=torch.bool)
        pad_mask[:, -2:] = True
        out = mixer.encode(x, pad_mask=pad_mask)
        assert out.pooled.shape == (2, cfg.d_model)
        assert out.hidden.shape[:2] == (2, cfg.seq_len)


def test_classifier_forward(manifest: MixerManifest) -> None:
    pytest.importorskip("torch")
    cfg = CommonConfig(d_model=16, seq_len=8, vocab_size=16)
    model = _MixerClassifier(cfg, MixerFamily.TRANSFORMER)
    ids = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len))
    logits = model(ids)
    assert logits.shape == (2, cfg.n_classes)


def test_dataset_deterministic() -> None:
    pytest.importorskip("torch")
    cfg = CommonConfig()
    ids1, labels1 = _make_dataset(cfg, seed=0, n=16)
    ids2, labels2 = _make_dataset(cfg, seed=0, n=16)
    assert torch.equal(ids1, ids2)
    assert torch.equal(labels1, labels2)


def test_run_fixture_campaign_produces_rows(manifest: MixerManifest) -> None:
    pytest.importorskip("torch")
    cfg = CommonConfig(
        n_train=32, n_eval=8, epochs=2, seeds=(0,), batch_size=8, d_model=16
    )
    report = run_fixture_campaign(
        manifest=MixerManifest(common_config=cfg, arms=manifest.arms),
        run_id="test_slm158",
    )
    assert report.status == "fixture"
    assert report.rows
    promotable_rows = [r for r in report.rows if r.promotable]
    assert promotable_rows
    assert report.version_stamp


def test_blocked_arm_row(manifest: MixerManifest) -> None:
    pytest.importorskip("torch")
    arms = list(manifest.arms)
    blocked = MixerArm(
        **{
            **arms[0].__dict__,
            "blocked": True,
            "blocker": "test blocker",
            "promotable": False,
        }
    )
    new_arms = [blocked if arm.arm_id == blocked.arm_id else arm for arm in arms]
    cfg = CommonConfig(
        n_train=32, n_eval=8, epochs=2, seeds=(0,), batch_size=8, d_model=16
    )
    report = run_fixture_campaign(
        manifest=MixerManifest(common_config=cfg, arms=tuple(new_arms)),
        run_id="test_blocked",
    )
    blocked_rows = [r for r in report.rows if r.arm_id == blocked.arm_id]
    assert blocked_rows
    assert blocked_rows[0].n_records == 0
    assert not blocked_rows[0].promotable


def test_report_row_from_dict() -> None:
    row = MixerReportRow(
        arm_id="A",
        family=MixerFamily.TRANSFORMER,
        seed=0,
        promotable=True,
        n_records=10,
        mean_loss=0.5,
        mean_accuracy=0.9,
        mean_latency_ms=1.0,
        param_count=100,
        notes=["note"],
    )
    data = row.to_dict()
    restored = MixerReportRow.from_dict(data)
    assert restored.arm_id == row.arm_id
    assert restored.family == row.family


def test_deterministic_replay(manifest: MixerManifest) -> None:
    pytest.importorskip("torch")
    cfg = CommonConfig(
        n_train=32, n_eval=8, epochs=2, seeds=(0,), batch_size=8, d_model=16
    )
    manifest = MixerManifest(common_config=cfg, arms=manifest.arms)
    report1 = run_fixture_campaign(manifest=manifest, run_id="replay1")
    report2 = run_fixture_campaign(manifest=manifest, run_id="replay2")
    rows1 = {f"{r.arm_id}-{r.seed}": r for r in report1.rows}
    rows2 = {f"{r.arm_id}-{r.seed}": r for r in report2.rows}
    assert set(rows1) == set(rows2)
    for key in rows1:
        assert rows1[key].mean_loss == pytest.approx(rows2[key].mean_loss, abs=1e-6)
        assert rows1[key].mean_accuracy == pytest.approx(
            rows2[key].mean_accuracy, abs=1e-6
        )
