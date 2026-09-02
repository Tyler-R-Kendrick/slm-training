"""Tests for the power_decidability preflight plugin."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_training.autoresearch.preflight.power_check import (
    CHECK,
    CONSERVATIVE_DEFAULT_SD,
    DEFAULT_LEDGER_PATH,
    MAX_REASONABLE_N,
    SD_SOURCE_UNMEASURED,
    PowerDecidabilityCheck,
    _pooled_ledger_sd,
)
from slm_training.autoresearch.screening_sample_size import MEASURED_PAIRED_SD


@pytest.fixture
def ledger_check(tmp_path: Path) -> PowerDecidabilityCheck:
    """The committed ledger with no expectations slot: pins the ledger source."""
    return PowerDecidabilityCheck(expectations_path=tmp_path / "no-expectations.json")


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
        assert verdict.verdict in {"pass", "warn", "block"}
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

    def test_adequately_powered_candidate_passes(
        self, ledger_check: PowerDecidabilityCheck
    ) -> None:
        pooled = _pooled_ledger_sd(DEFAULT_LEDGER_PATH)
        assert pooled is not None
        from slm_training.autoresearch.power import required_n_for_effect

        enough = max(8, required_n_for_effect(0.05, pooled[0], 0.05, paired=True))
        verdict = ledger_check.run(_candidate(n_seeds=enough, minimum_effect=0.05))
        assert verdict.verdict == "pass"
        assert verdict.data["sd_measured"] is True
        assert verdict.data["min_attainable_p"] == pytest.approx(2 / 2**enough)

    def test_still_accumulating_candidate_warns_not_blocks(self) -> None:
        # Same reachable design as the n=8 pass case, but not there yet: the
        # seeds-policy fix (WP-3 reconciliation) must not block an arm that
        # simply hasn't accumulated enough cycles yet — that would prevent
        # it from ever accumulating past this cycle.
        verdict = CHECK.run(_candidate(n_seeds=2, minimum_effect=0.05))
        assert verdict.verdict == "warn"
        assert verdict.data["required_n_for_effect"] <= MAX_REASONABLE_N
        assert any("still accumulating" in reason for reason in verdict.reasons)

    def test_structurally_undecidable_design_still_blocks(self) -> None:
        # RC1's own design (n_seeds=3, minimum_effect=0.01) requires far more
        # than MAX_REASONABLE_N seeds — genuinely undecidable, not merely
        # early in accumulation.
        verdict = CHECK.run(_candidate())
        assert verdict.verdict == "block"
        assert verdict.data["required_n_for_effect"] > MAX_REASONABLE_N
        assert any("realistic accumulation" in reason for reason in verdict.reasons)

    def test_process_arm_warns_instead_of_blocking(self) -> None:
        # First-train I10 / heal_resume is execution, not a confirmatory claim.
        verdict = CHECK.run(_candidate(n_seeds=1, process_arm=True, heal_resume=True))
        assert verdict.verdict == "warn"
        assert verdict.data.get("process_arm") is True
        assert any("not a confirmatory design" in reason for reason in verdict.reasons)

    def test_ledger_sd_used_for_primary_metric(
        self, ledger_check: PowerDecidabilityCheck
    ) -> None:
        verdict = ledger_check.run(_candidate(n_seeds=8, minimum_effect=0.05))
        assert "evidence_ledger" in verdict.data["sd_source"]
        assert verdict.data["sd_measured"] is True
        pooled = _pooled_ledger_sd(DEFAULT_LEDGER_PATH)
        assert pooled is not None
        assert verdict.data["sd"] == pytest.approx(pooled[0])

    def test_foreign_metric_is_unmeasured_and_warns(self) -> None:
        # No paired SD exists for this metric: the conservative default is
        # reported as an advisory number only, never grounds for a block.
        verdict = CHECK.run(
            _candidate(endpoint_metric="held_out.ast_beq_rate", n_seeds=8)
        )
        assert verdict.data["sd"] == pytest.approx(CONSERVATIVE_DEFAULT_SD)
        assert verdict.data["sd_source"] == SD_SOURCE_UNMEASURED
        assert verdict.data["sd_measured"] is False
        assert verdict.verdict == "warn"

    def test_missing_ledger_falls_back_to_tagged_constant_for_own_metric(
        self, tmp_path: Path
    ) -> None:
        check = PowerDecidabilityCheck(
            ledger_path=tmp_path / "missing.json",
            expectations_path=tmp_path / "missing-expectations.json",
        )
        verdict = check.run(_candidate(n_seeds=8, minimum_effect=0.05))
        # smoke.structural_similarity owns MEASURED_PAIRED_SD; a measured SD
        # that demands n=96 > MAX_REASONABLE_N is a genuine block.
        assert verdict.data["sd"] == pytest.approx(MEASURED_PAIRED_SD)
        assert "measured_constant" in verdict.data["sd_source"]
        assert verdict.data["sd_measured"] is True
        assert verdict.verdict == "block"


class TestRC2UnmeasuredMetric:
    """RC2: a confirmatory smoke.eval_nll arm must never block on a borrowed SD."""

    @pytest.mark.parametrize("n_seeds", [1, 2, 3, 8])
    def test_confirmatory_eval_nll_arm_warns_not_blocks(self, n_seeds: int) -> None:
        verdict = CHECK.run(
            _candidate(
                endpoint_metric="smoke.eval_nll",
                minimum_effect=0.02,
                n_seeds=n_seeds,
                claim_class="diagnostic",
            )
        )
        assert verdict.verdict in {"pass", "warn"}
        assert verdict.data["sd_source"] == SD_SOURCE_UNMEASURED
        assert verdict.data["sd_measured"] is False
        joined = " ".join(verdict.reasons)
        assert "unmeasured" in joined
        assert "observed_paired_sd_by_metric[eval_nll]" in joined

    def test_recorded_expectations_sd_makes_eval_nll_measured(
        self, tmp_path: Path
    ) -> None:
        expectations = tmp_path / "metric_expectations.screening.v1.json"
        expectations.write_text(
            json.dumps({"observed_paired_sd_by_metric": {"eval_nll": 0.03}}),
            encoding="utf-8",
        )
        check = PowerDecidabilityCheck(
            ledger_path=tmp_path / "missing.json", expectations_path=expectations
        )
        verdict = check.run(
            _candidate(endpoint_metric="smoke.eval_nll", minimum_effect=0.02, n_seeds=8)
        )
        assert verdict.data["sd_measured"] is True
        assert "metric_expectations" in verdict.data["sd_source"]
        assert verdict.data["sd"] == pytest.approx(0.03)
        assert verdict.verdict in {"pass", "warn", "block"}
        # With a measured SD the required n is a real determination again.
        from slm_training.autoresearch.power import required_n_for_effect

        assert verdict.data["required_n_for_effect"] == required_n_for_effect(
            0.02, 0.03, 0.05, paired=True
        )

    def test_unmeasured_never_blocks_even_when_required_n_is_absurd(self) -> None:
        verdict = CHECK.run(
            _candidate(endpoint_metric="smoke.eval_nll", minimum_effect=0.001, n_seeds=1)
        )
        assert verdict.data["required_n_for_effect"] > MAX_REASONABLE_N
        assert verdict.verdict == "warn"

    def test_empty_endpoint_metric_blocks_as_malformed(self) -> None:
        verdict = CHECK.run(_candidate(endpoint_metric=""))
        assert verdict.verdict == "block"
        assert any("endpoint_metric" in reason for reason in verdict.reasons)


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
        check = PowerDecidabilityCheck(
            ledger_path=bad, expectations_path=tmp_path / "missing-expectations.json"
        )
        verdict = check.run(_candidate(n_seeds=8, minimum_effect=0.05))
        assert "measured_constant" in verdict.data["sd_source"]
        corrupt_expectations = tmp_path / "metric_expectations.screening.v1.json"
        corrupt_expectations.write_text("{not json", encoding="utf-8")
        check = PowerDecidabilityCheck(
            ledger_path=bad, expectations_path=corrupt_expectations
        )
        verdict = check.run(_candidate(endpoint_metric="smoke.eval_nll", n_seeds=8))
        assert verdict.data["sd_source"] == SD_SOURCE_UNMEASURED
        assert verdict.verdict == "warn"

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
