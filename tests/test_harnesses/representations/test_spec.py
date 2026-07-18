"""LDI4-02 torch-free schema: capture manifest, SAE config, arms, train-only selection."""

from __future__ import annotations

import pytest

from slm_training.harnesses.preference.local_decisions import split_for_group
from slm_training.harnesses.representations.spec import (
    CaptureManifest,
    CaptureRow,
    FeatureSelectionError,
    SAEConfig,
    matched_sae_arms,
    select_features_train_only,
)


def _row(group_id: str, state_id: str, role: str = "target") -> CaptureRow:
    return CaptureRow(
        state_id=state_id,
        group_id=group_id,
        split=split_for_group(group_id),
        architecture="twotower",
        policy_checkpoint_sha="ckpt",
        tokenizer_sha="tok",
        decode_config_hash="dec",
        verifier_bundle_hash="ver",
        site="denoiser.block.3.residual",
        position="exact_decision",
        hidden_size=16,
        dtype="float32",
        activation_content_hash=f"hash-{state_id}",
        role=role,
    )


def test_capture_row_split_must_match_group():
    good = _row("group-a", "s0")
    assert good.split == split_for_group("group-a")
    with pytest.raises(ValueError):
        CaptureRow(**{**good.to_dict(), "split": ("held_out" if good.split == "train" else "train")})
    with pytest.raises(ValueError):
        CaptureRow(**{**good.to_dict(), "role": "bogus"})


def test_capture_manifest_rejects_train_held_overlap():
    # find two groups on opposite splits sharing a state id -> leakage
    groups = [f"g{i}" for i in range(40)]
    train_g = next(g for g in groups if split_for_group(g) == "train")
    held_g = next(g for g in groups if split_for_group(g) == "held_out")
    rows = (_row(train_g, "dup"), _row(held_g, "dup"))
    with pytest.raises(ValueError):
        CaptureManifest(site="denoiser.block.3.residual", hidden_size=16, rows=rows)


def test_capture_manifest_identity_fingerprint_is_stable_and_site_consistent():
    groups = [f"g{i}" for i in range(40)]
    train_g = next(g for g in groups if split_for_group(g) == "train")
    m = CaptureManifest("denoiser.block.3.residual", 16, (_row(train_g, "s1"),))
    assert m.identity_fingerprint() == CaptureManifest("denoiser.block.3.residual", 16, (_row(train_g, "s1"),)).identity_fingerprint()
    with pytest.raises(ValueError):
        CaptureManifest("denoiser.block.3.residual", 32, (_row(train_g, "s1"),))  # width mismatch


def test_sae_config_fail_closed_and_derived_width():
    cfg = SAEConfig(d_in=8, expansion_factor=4)
    assert cfg.dict_width == 32
    assert cfg.fingerprint() == SAEConfig(d_in=8, expansion_factor=4).fingerprint()
    with pytest.raises(ValueError):
        SAEConfig(d_in=0)
    with pytest.raises(ValueError):
        SAEConfig(d_in=8, nonlinearity="gelu")
    with pytest.raises(ValueError):
        SAEConfig(d_in=8, lambda_sparse=-1.0)
    with pytest.raises(ValueError):
        SAEConfig.from_mapping({"d_in": 8, "bogus": 1})


def test_matched_arms_cover_s0_to_s7_and_are_train_only():
    arms = matched_sae_arms(site="denoiser.block.3.residual")
    ids = [a.arm_id for a in arms]
    assert ids == ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7"]
    # every supervised steering arm selects on train groups only.
    for a in arms:
        if a.method not in ("no_intervention", "random_normalized_direction"):
            assert a.selection_data == "train_only", a.arm_id
    # SAE arms carry no free parameter advantage recorded as unbounded.
    assert {a.arm_id for a in arms if a.method.startswith("top") or "sparse" in a.method} == {"S6", "S7"}


def test_feature_selection_is_train_only_fail_closed():
    assert select_features_train_only({"train": [0.1, 0.9, 0.5]}, top_k=2) == (1, 2)
    with pytest.raises(FeatureSelectionError):
        select_features_train_only({"held_out": [0.9, 0.1]}, top_k=1)
    with pytest.raises(FeatureSelectionError):
        select_features_train_only({"train": []}, top_k=1)
