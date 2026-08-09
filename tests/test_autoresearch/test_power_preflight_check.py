"""Tests for the power_decidability preflight plugin."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_training.autoresearch.preflight.power_check import (
    CHECK,
    CONSERVATIVE_DEFAULT_SD,
    DEFAULT_LEDGER_PATH,
    PowerDecidabilityCheck,
    _pooled_ledger_sd,
)


def _candidate(**overrides: object) -> dict:
    base: dict = {
        "hypothesis_text": "closing containers earlier improves similarity",
        "lever_keys": ["decode.container_close"],
        "config_fingerprint": "deadbeef",
        "n_seeds": 3,
        "steps": 22,
        "minimum_effect": 0.01,
        "endpoint_metric": "smoke.structural_similarity",
    }
    base.update(overrides)
    return base


class TestPluginSurface:
    def test_module_level_check_object(self) -> None:
        assert CHECK.check_id == "power_decidability"
        assert callable(CHECK.run)

    def test_verdict_is_structural(self) -> None:
        verdict = CHECK.run(_candidate())
        assert verdict.check_id == "power_decidability"
        assert verdict.verdict in {"pass", "block"}
        assert isinstance(verdict.reasons, list)


class TestVerdicts:
    def test_rc1_screening_candidate_blocks(self) -> None:
        # n_seeds=3 at minimum_effect=0.01 is RC1's undecidable design.
        verdict = CHECK.run(_candidate())
        assert verdict.verdict == "block"
        joined = " ".join(verdict.reasons)
        assert "min_attainable_p=0.25" in joined
        assert "required_n_for_effect" in joined

    def test_block_details_support_replanning(self) -> None:
        verdict = CHECK.run(_candidate())
        details = verdict.data
        assert details["min_attainable_p"] == pytest.approx(0.25)
        assert details["required_n_for_effect"] >= 6
        assert details["n_seeds"] == 3
        assert details["sd"] > 0

    def test_adequately_powered_candidate_passes(self) -> None:
        verdict = CHECK.run(_candidate(n_seeds=8, minimum_effect=0.05))
        assert verdict.verdict == "pass"
        assert verdict.data["min_attainable_p"] == pytest.approx(2 / 2**8)

    def test_ledger_sd_used_for_primary_metric(self) -> None:
        verdict = CHECK.run(_candidate(n_seeds=8, minimum_effect=0.05))
        assert "evidence_ledger" in verdict.data["sd_source"]
        pooled = _pooled_ledger_sd(DEFAULT_LEDGER_PATH)
        assert pooled is not None
        assert verdict.data["sd"] == pytest.approx(pooled[0])

    def test_foreign_metric_uses_conservative_default(self) -> None:
        verdict = CHECK.run(
            _candidate(endpoint_metric="held_out.ast_beq_rate", n_seeds=8)
        )
        assert verdict.data["sd"] == pytest.approx(CONSERVATIVE_DEFAULT_SD)
        assert "conservative default" in verdict.data["sd_source"]

    def test_missing_ledger_falls_back_to_conservative_default(
        self, tmp_path: Path
    ) -> None:
        check = PowerDecidabilityCheck(ledger_path=tmp_path / "missing.json")
        verdict = check.run(_candidate(n_seeds=8, minimum_effect=0.05))
        assert verdict.data["sd"] == pytest.approx(CONSERVATIVE_DEFAULT_SD)
        # Conservative default inflates the MDE, so this design now blocks.
        assert verdict.verdict == "block"


class TestNeverRaises:
    @pytest.mark.parametrize(
        "candidate",
        [
            None,
            [],
            "screening",
            {},
            {"n_seeds": "three"},
            {"n_seeds": 0},
            {"n_seeds": -2},
            {"n_seeds": True},
            {"n_seeds": 3.5},
            _candidate(minimum_effect="tiny"),
            _candidate(minimum_effect=float("nan")),
            _candidate(endpoint_metric=None),
        ],
    )
    def test_malformed_candidates_block_without_raising(
        self, candidate: object
    ) -> None:
        verdict = CHECK.run(candidate)  # type: ignore[arg-type]
        assert verdict.check_id == "power_decidability"
        assert verdict.reasons
        # Every malformed or undecidable candidate fails closed.
        assert verdict.verdict == "block"

    def test_missing_minimum_effect_uses_policy_default(self) -> None:
        candidate = _candidate()
        del candidate["minimum_effect"]
        verdict = CHECK.run(candidate)
        assert verdict.data["minimum_effect"] == pytest.approx(0.01)
        assert any("screening default" in reason for reason in verdict.reasons)

    def test_corrupt_ledger_never_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "evidence_ledger.v1.json"
        bad.write_text("{not json", encoding="utf-8")
        check = PowerDecidabilityCheck(ledger_path=bad)
        verdict = check.run(_candidate(n_seeds=8, minimum_effect=0.05))
        assert "conservative default" in verdict.data["sd_source"]

    def test_ledger_with_insufficient_dof_rejected(self, tmp_path: Path) -> None:
        thin = tmp_path / "evidence_ledger.v1.json"
        thin.write_text(
            json.dumps({"arms": {"a": {"m2_delta": 0.1, "n_delta": 3}}}),
            encoding="utf-8",
        )
        assert _pooled_ledger_sd(thin) is None


class TestPackageDiscovery:
    def test_run_preflight_discovers_this_check(self) -> None:
        from slm_training.autoresearch.preflight import has_block, run_preflight

        verdicts = run_preflight(_candidate())
        mine = [v for v in verdicts if v.check_id == "power_decidability"]
        assert len(mine) == 1
        assert mine[0].verdict == "block"
        assert has_block(verdicts)


class TestPooledSd:
    def test_matches_hand_computed_pooled_variance(self, tmp_path: Path) -> None:
        ledger = tmp_path / "evidence_ledger.v1.json"
        arms = {
            f"arm-{i}": {"m2_delta": 0.04, "n_delta": 6} for i in range(6)
        }
        ledger.write_text(json.dumps({"arms": arms}), encoding="utf-8")
        pooled = _pooled_ledger_sd(ledger)
        assert pooled is not None
        sd, dof = pooled
        assert dof == 30
        assert sd == pytest.approx((6 * 0.04 / 30) ** 0.5)

    def test_repo_ledger_is_usable(self) -> None:
        pooled = _pooled_ledger_sd(DEFAULT_LEDGER_PATH)
        assert pooled is not None
        sd, dof = pooled
        assert sd > 0
        assert dof >= 20
