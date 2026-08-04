"""Fixture campaign runner for the advisory semantic-factor frontier."""

from __future__ import annotations

from slm_training.harnesses.experiments.anti_e237_semantic_factor_frontier import (
    build_campaign,
    build_ranked_suite,
    lock_campaign,
    run_semantic_factor_frontier,
)
from slm_training.harnesses.experiments.semantic_factor_claims import evaluate_claims


def test_campaign_locks_before_outcomes() -> None:
    campaign = build_campaign(source_commit="a" * 40, source_dirty=False)
    lock = lock_campaign(campaign)
    assert lock.manifest_sha256
    assert lock.manifest.claim_class == "fixture"
    assert any(a.role == "control" for a in lock.manifest.arms)
    assert campaign.requires_rl is False


def test_ranked_suite_has_enough_ambiguous_cases() -> None:
    suite = build_ranked_suite()
    ranked = [e for e in suite if e.get("family") in {"ranked", "adversarial"}]
    assert len(ranked) >= 24
    for e in ranked:
        assert e["coverage"] == "complete"
        assert len(e["legal"]) > 1
        assert e["gold"] in e["legal"]
        # Baseline argmax is not gold (so residual can improve).
        base = max(e["legal"], key=lambda c: (e["baseline"][c], c))
        assert base != e["gold"], e["example_id"]


def test_fixture_run_preserves_invariants_and_claims(tmp_path) -> None:
    payload = run_semantic_factor_frontier(
        seeds=(0,),
        out_dir=tmp_path,
        source_commit="b" * 40,
        source_dirty=False,
    )
    assert payload["claim_class"] == "fixture"
    assert payload["membership_roundtrip_ok"] is True
    assert payload["decode_off_identical_to_control"] is True
    assert payload["support_preservation_ok"] is True
    assert payload["n_ranked_per_arm"] >= 24
    assert payload["math_properties"]["column_stochastic_ok"] is True
    assert payload["math_properties"]["soft_token_collision_ok"] is True
    assert payload["unimplemented"]["rl"] is True
    assert payload["unimplemented"]["spectral_training"] is True
    assert payload["unimplemented"]["production_topology_authority"] is True
    assert payload["unimplemented"]["recurrent_semantic_inference"] is True
    for arm in payload["arms"]:
        assert "singleton_work_performed" not in arm["kill_criteria_hit"]
        assert arm["singleton_skips"] >= 1
        assert "mrr" in arm
        assert "ranking_accuracy_ci95" in arm
        # Runtime is first-class.
        assert arm["wall_ms_mean"] >= 0.0
        assert arm["wall_ms_p50"] >= 0.0
        assert arm["wall_ms_p95"] >= arm["wall_ms_p50"] - 1e-9
        assert arm["wall_ms_total"] > 0.0
        assert "wall_ms_mean_vs_control" in arm
        assert "quality_per_ms" in arm
        assert "delta_accuracy_per_extra_ms" in arm
    assert payload["runtime"]["timer"] == "time.perf_counter"
    assert payload["runtime"]["campaign_wall_s"] > 0.0
    claims = {c["claim_id"]: c for c in payload["claims"]}
    assert claims["SFF-C2-singleton-zero-work"]["verdict"] == "validated"
    assert claims["SFF-C3-unknown-preserved"]["verdict"] == "validated"
    assert claims["SFF-C4-decode-off-identity"]["verdict"] == "validated"
    assert claims["SFF-C5-factor-node-lossless"]["verdict"] == "validated"
    assert claims["SFF-C14-soft-token-not-injective"]["verdict"] == "validated"
    assert claims["SFF-C15-no-rl-no-spectral"]["verdict"] == "validated"
    assert claims["SFF-C17-runtime-tracked"]["verdict"] == "validated"
    # Fixture cannot pass promotion bar.
    assert claims["SFF-C16-anti-e237-promotion-bar"]["verdict"] == "invalidated"
    # Recompute claims from payload matches.
    recomputed = evaluate_claims(payload)
    assert len(recomputed) == len(payload["claims"])
    assert (tmp_path / "scoreboard.json").is_file()
    assert (tmp_path / "claims.json").is_file()
