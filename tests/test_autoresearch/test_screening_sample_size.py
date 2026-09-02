"""screening_sample_size/v1: certified screening-n range for the climb loop."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import pytest
from pydantic import ValidationError

from slm_training.autoresearch.power import min_attainable_n, required_n_for_effect
from slm_training.autoresearch.schemas import DEFAULT_ALLOWED_KNOBS
from slm_training.autoresearch.power import min_detectable_effect
from slm_training.autoresearch.screening_sample_size import (
    DEFAULT_SCREENING_EXPECTATIONS_PATH,
    FINDING_POWER_SD_UNMEASURED,
    FINDING_RANGE_EMPTY,
    FROZEN_EVAL_SNAPSHOTS,
    FrozenEvalSnapshotError,
    MEASURED_PAIRED_SD,
    MEASURED_PAIRED_SD_METRIC,
    OBSERVED_PAIRED_SD_BY_METRIC_KEY,
    SCREENING_SAMPLE_SIZE_SCHEMA,
    SCREENING_SMOKE6_EVAL_VERSION,
    ScreeningSampleSizeObservation,
    TARGET_SMOKE_N,
    assert_eval_publish_target_writable,
    compute_screening_sample_size,
    extra_smoke_fixtures_for_deficit,
    lookup_paired_sd_for_metric,
    metric_leaf,
)


def _obs(**overrides: object) -> ScreeningSampleSizeObservation:
    base = {
        "suite_records": 24,
        "arm_wall_seconds": 70,
        "min_train_floor_seconds": 20,
        "suite_overhead_seconds": 8,
        "per_record_decode_floor_seconds": 2,
    }
    base.update(overrides)
    return ScreeningSampleSizeObservation(**base)


def test_extra_smoke_fixtures_cover_deficit_without_duplicates() -> None:
    # Deficit records come from the certified corpus' validation bucket, in
    # the test_seeds.jsonl shape, never from a hand-written tuple.
    extras = extra_smoke_fixtures_for_deficit(existing_ids=set(), need=3)
    assert len(extras) == 3
    for row in extras:
        assert row["split"] == "smoke"
        assert row["meta"]["suite"] == "smoke"
        assert row["meta"]["certified_corpus"] == "openui_verified_v1"
        assert row["meta"]["root_family_split"] == "validation"
        assert row["openui"] and row["prompt"]
    used = {row["id"] for row in extras}
    assert len(used) == 3
    # Deterministic for the same corpus, seed and exclusion set.
    again = extra_smoke_fixtures_for_deficit(existing_ids=set(), need=3)
    assert {row["id"] for row in again} == used
    more = extra_smoke_fixtures_for_deficit(existing_ids=set(used), need=3)
    assert more
    assert used.isdisjoint({row["id"] for row in more})
    exhausted = extra_smoke_fixtures_for_deficit(
        existing_ids={row["id"] for row in extras + more}, need=0
    )
    assert exhausted == []


def test_extra_smoke_fixtures_can_fill_target_from_certified_pool() -> None:
    seeded = {"smoke_hero_01", "smoke_button_01", "smoke_callout_01"}
    existing = set(seeded)
    need = 3 * TARGET_SMOKE_N
    extras = extra_smoke_fixtures_for_deficit(existing_ids=existing, need=need)
    ids = [row["id"] for row in extras]
    assert len(extras) == need  # the old 21-tuple could never reach this
    assert len(set(ids)) == len(ids)
    assert seeded.isdisjoint(ids)
    assert existing == seeded | set(ids)  # updated in place for the next call
    assert len({row["source"] for row in extras}) > 1  # stratified by source


def test_latency_probe_knobs_are_allowed() -> None:
    assert "latency_probe_records" in DEFAULT_ALLOWED_KNOBS
    assert "latency_probe_planned_n" in DEFAULT_ALLOWED_KNOBS


def test_decidability_floor_matches_power_module_exact_search() -> None:
    for alpha, expected in (("1/4", 3), ("1/10", 5), ("1/20", 6)):
        report = compute_screening_sample_size(_obs(alpha=alpha))
        assert report.decidability_floor_n == expected
        assert report.decidability_floor_n == min_attainable_n(
            float(Fraction(alpha)), paired=True
        )


def test_budget_ceiling_arithmetic() -> None:
    assert (
        compute_screening_sample_size(_obs(per_record_decode_floor_seconds=14))
        .budget_ceiling_n
        == 3
    )
    assert compute_screening_sample_size(_obs()).budget_ceiling_n == 21
    assert (
        compute_screening_sample_size(_obs(arm_wall_seconds=10)).budget_ceiling_n == 0
    )


def test_feasible_range_climbs_at_smallest_sufficient_n() -> None:
    report = compute_screening_sample_size(_obs())
    assert report.verdict == "feasible"
    assert report.n_min == 6
    assert report.chosen_n == 6
    assert report.n_max == 21
    assert report.binding_constraints == ()
    assert report.findings == ()


def test_today_fixture_ceiling_is_suite_bound() -> None:
    """The committed 3-record smoke suite makes the certified range empty."""

    report = compute_screening_sample_size(_obs(suite_records=3))
    assert report.verdict == "infeasible_range_empty"
    assert report.n_min == 6
    assert report.chosen_n is None
    assert "suite_volume" in report.binding_constraints
    finding = next(f for f in report.findings if f["code"] == FINDING_RANGE_EMPTY)
    assert finding["authority"] == "climb_signal_not_gate"
    assert report.must_generate is True
    assert "do not screen" in finding["suggestion"]


def test_wall_budget_binding_when_decode_is_expensive() -> None:
    report = compute_screening_sample_size(
        _obs(suite_records=24, per_record_decode_floor_seconds=14)
    )
    assert report.verdict == "infeasible_range_empty"
    assert report.budget_ceiling_n == 3
    assert report.binding_constraints == ("wall_budget",)
    assert report.must_generate is False


def test_both_axes_binding() -> None:
    report = compute_screening_sample_size(
        _obs(suite_records=3, per_record_decode_floor_seconds=14)
    )
    assert report.verdict == "infeasible_range_empty"
    assert set(report.binding_constraints) == {"wall_budget", "suite_volume"}


def test_insufficient_evidence_without_decode_observation() -> None:
    report = compute_screening_sample_size(
        _obs(per_record_decode_floor_seconds=None)
    )
    assert report.verdict == "insufficient_evidence"
    assert report.budget_ceiling_n is None
    assert report.chosen_n is None
    assert report.n_min == 6  # the exact floor is still computed


def test_power_floor_mde_is_monotone_decreasing_in_n() -> None:
    sd = 0.1741
    mdes = [
        min_detectable_effect(n, sd, 0.05, 0.8, paired=True)
        for n in (6, 12, 24, 48)
    ]
    assert mdes == sorted(mdes, reverse=True)
    floors = [
        compute_screening_sample_size(
            _obs(minimum_effect=str(mdes[0]), observed_sd=str(sd))
        ).power_floor_n
    ]
    # Larger declared effect cannot raise the power floor.
    smaller = compute_screening_sample_size(
        _obs(minimum_effect=str(mdes[-1]), observed_sd=str(sd))
    )
    larger = compute_screening_sample_size(
        _obs(minimum_effect=str(mdes[0]), observed_sd=str(sd))
    )
    assert smaller.power_floor_n is not None
    assert larger.power_floor_n is not None
    assert smaller.power_floor_n >= larger.power_floor_n
    assert floors[0] == larger.power_floor_n


def test_chosen_n_is_max_of_decidability_and_power_floors() -> None:
    report = compute_screening_sample_size(
        _obs(minimum_effect="1/100", observed_sd="1/10", suite_records=64)
    )
    assert report.decidability_floor_n == 6
    assert report.power_floor_n is not None
    assert report.n_min == max(report.decidability_floor_n, report.power_floor_n)


def test_published_smoke24_is_disjoint_and_does_not_mutate_frozen() -> None:
    from pathlib import Path

    from slm_training.dsl.schema import load_jsonl

    published = Path(
        "src/slm_training/resources/data/eval/e938_role_safe_all_targets_smoke24_v1"
    )
    frozen = Path(
        "src/slm_training/resources/data/eval/e938_role_safe_all_targets_smoke6_v1"
    )
    smoke = list(load_jsonl(published / "suites" / "smoke" / "records.jsonl"))
    frozen_smoke = list(load_jsonl(frozen / "suites" / "smoke" / "records.jsonl"))
    assert len(smoke) >= TARGET_SMOKE_N
    assert len(frozen_smoke) == 6
    assert frozen.is_dir()
    train_ids = {
        rec.id
        for rec in load_jsonl(
            Path("src/slm_training/resources/data/train/wf_smoke_v2/records.jsonl")
        )
    }
    assert train_ids.isdisjoint({rec.id for rec in smoke})


def test_publish_refuses_frozen_eval_dirs() -> None:
    with pytest.raises(FrozenEvalSnapshotError):
        assert_eval_publish_target_writable(SCREENING_SMOKE6_EVAL_VERSION)
    for frozen in FROZEN_EVAL_SNAPSHOTS:
        with pytest.raises(FrozenEvalSnapshotError):
            assert_eval_publish_target_writable(frozen)
    assert_eval_publish_target_writable("e938_role_safe_all_targets_smoke24_v1")


def test_power_floor_is_assumption_backed_and_dominates() -> None:
    report = compute_screening_sample_size(
        _obs(minimum_effect="1/100", observed_sd="1/10")
    )
    expected = required_n_for_effect(0.01, 0.1, 0.05, paired=True)
    assert report.power_floor_n == expected
    assert report.n_min == max(6, expected)
    bound = next(b for b in report.bounds if b.bound_ast_id.startswith("power."))
    assert bound.authority == "assumption_backed"
    exact = next(
        b for b in report.bounds if b.bound_ast_id.startswith("bound.screening_n")
    )
    assert exact.authority == "theorem_backed_exact"


def test_floor_beyond_search_cap_fails_closed() -> None:
    report = compute_screening_sample_size(_obs(max_candidate_n=4))
    assert report.decidability_floor_n is None
    assert report.verdict == "infeasible_range_empty"
    assert report.chosen_n is None
    assert any(
        f["code"] == "screening_n_floor_beyond_search_cap" for f in report.findings
    )


def test_strict_validation() -> None:
    with pytest.raises(ValidationError):
        _obs(alpha="0")
    with pytest.raises(ValidationError):
        _obs(alpha="1")
    # Effect without sd is the *unmeasured* power case (valid, advisory).
    unmeasured = _obs(minimum_effect="1/100")
    assert unmeasured.observed_sd is None
    with pytest.raises(ValidationError):
        _obs(observed_sd="1/10")  # sd without effect
    with pytest.raises(ValidationError):
        _obs(minimum_effect="1/100", observed_sd="-1/10")
    with pytest.raises(ValidationError):
        _obs(unknown_field=1)


# ---------------------------------------------------------------------------
# RC1: unmeasured same-metric SD never borrows another metric's variance
# ---------------------------------------------------------------------------


def test_unmeasured_sd_reports_unmeasured_power_floor_and_affordable_n() -> None:
    # Policy declares an effect for smoke.eval_nll; no paired SD exists for
    # that metric. The exact floor (6) stands alone; the screen spends the
    # affordable n as an advisory (insufficient_evidence) verdict.
    report = compute_screening_sample_size(
        _obs(
            minimum_effect="1/50",
            observed_sd_source="unmeasured",
            observed_sd_metric="smoke.eval_nll",
        )
    )
    assert report.power_floor_status == "unmeasured"
    assert report.power_floor_n is None
    assert report.n_min == 6
    assert report.verdict == "insufficient_evidence"
    assert report.budget_ceiling_n == 21
    assert report.chosen_n == 21  # min(suite=24, budget=21, cap=64)
    assert report.must_generate is False
    assert report.binding_constraints == ()
    assert report.observed_sd_source == "unmeasured"
    assert report.observed_sd_metric == "smoke.eval_nll"
    assert any(f["code"] == FINDING_POWER_SD_UNMEASURED for f in report.findings)
    assert not any(b.bound_ast_id.startswith("power.") for b in report.bounds)


def test_unmeasured_sd_caps_affordable_n_at_search_cap_and_suite() -> None:
    wide = compute_screening_sample_size(
        _obs(minimum_effect="1/50", arm_wall_seconds=180, suite_records=24)
    )
    assert wide.chosen_n == 24  # suite binds before the 76-record wall budget
    capped = compute_screening_sample_size(
        _obs(
            minimum_effect="1/50",
            arm_wall_seconds=400,
            suite_records=200,
            max_candidate_n=32,
        )
    )
    assert capped.chosen_n == 32


def test_unmeasured_sd_must_generate_only_below_exact_floor() -> None:
    thin = compute_screening_sample_size(
        _obs(minimum_effect="1/50", suite_records=3)
    )
    assert thin.verdict == "infeasible_range_empty"
    assert thin.must_generate is True
    assert thin.n_min == 6
    enough = compute_screening_sample_size(
        _obs(minimum_effect="1/50", suite_records=6)
    )
    assert enough.verdict == "insufficient_evidence"
    assert enough.chosen_n == 6
    assert enough.must_generate is False


def test_measured_sd_still_yields_power_floor() -> None:
    report = compute_screening_sample_size(
        _obs(
            minimum_effect="1/20",
            observed_sd="1741/10000",
            observed_sd_source="measured_constant",
            observed_sd_metric="smoke.structural_similarity",
        )
    )
    assert report.power_floor_status == "measured"
    assert report.power_floor_n == required_n_for_effect(
        0.05, 0.1741, 0.05, paired=True
    )
    assert report.n_min == report.power_floor_n == 96
    assert report.verdict == "infeasible_range_empty"
    bound = next(b for b in report.bounds if b.bound_ast_id.startswith("power."))
    assert bound.env["observed_sd_metric"] == "smoke.structural_similarity"


def test_metric_leaf_and_matching() -> None:
    assert metric_leaf("smoke.eval_nll") == "eval_nll"
    assert metric_leaf("eval_nll") == "eval_nll"
    assert metric_leaf("") == ""


def test_lookup_prefers_expectations_slot_then_ledger_then_constant(
    tmp_path: Path,
) -> None:
    expectations = tmp_path / "metric_expectations.screening.v1.json"
    ledger = tmp_path / "evidence_ledger.v1.json"
    # 1. Nothing recorded anywhere for eval_nll -> unmeasured (never borrowed).
    expectations.write_text(json.dumps({"observed_paired_sd_by_metric": {}}))
    ledger.write_text(
        json.dumps({"arms": {f"a{i}": {"m2_delta": 0.04, "n_delta": 6} for i in range(6)}})
    )
    miss = lookup_paired_sd_for_metric(
        "smoke.eval_nll", expectations_path=expectations, ledger_path=ledger
    )
    assert miss.measured is False
    assert miss.source == "unmeasured"
    assert miss.observed_sd is None
    # The same untagged ledger *is* evidence for the legacy structural primary.
    legacy = lookup_paired_sd_for_metric(
        "smoke.structural_similarity", expectations_path=expectations, ledger_path=ledger
    )
    assert legacy.source == "evidence_ledger"
    assert float(Fraction(legacy.observed_sd)) == pytest.approx((6 * 0.04 / 30) ** 0.5)
    # ...but not for the held-out suite of the same leaf.
    held = lookup_paired_sd_for_metric(
        "held_out.structural_similarity", expectations_path=expectations, ledger_path=ledger
    )
    assert held.source == "unmeasured"
    # 2. A metric-keyed ledger counts for its own metric.
    ledger.write_text(
        json.dumps(
            {
                "metric": "smoke.eval_nll",
                "arms": {f"a{i}": {"m2_delta": 0.09, "n_delta": 6} for i in range(6)},
            }
        )
    )
    keyed = lookup_paired_sd_for_metric(
        "smoke.eval_nll", expectations_path=expectations, ledger_path=ledger
    )
    assert keyed.source == "evidence_ledger"
    assert float(Fraction(keyed.observed_sd)) == pytest.approx((6 * 0.09 / 30) ** 0.5)
    # 3. The expectations slot wins over the ledger (leaf key, object value).
    expectations.write_text(
        json.dumps(
            {
                "observed_paired_sd_by_metric": {
                    "eval_nll": {"sd": "3/100", "n_deltas": 405, "source": "test"}
                }
            }
        )
    )
    slot = lookup_paired_sd_for_metric(
        "smoke.eval_nll", expectations_path=expectations, ledger_path=ledger
    )
    assert slot.source == "metric_expectations"
    assert slot.observed_sd == "3/100"
    assert slot.detail["key"] == "eval_nll"
    # 4. Missing files degrade to the tagged constant for its own metric only.
    const = lookup_paired_sd_for_metric(
        "smoke.structural_similarity",
        expectations_path=tmp_path / "none.json",
        ledger_path=tmp_path / "none-ledger.json",
    )
    assert const.source == "measured_constant"
    assert float(Fraction(const.observed_sd)) == pytest.approx(MEASURED_PAIRED_SD)
    assert MEASURED_PAIRED_SD_METRIC == "smoke.structural_similarity"
    none = lookup_paired_sd_for_metric(
        "smoke.eval_nll",
        expectations_path=tmp_path / "none.json",
        ledger_path=tmp_path / "none-ledger.json",
    )
    assert none.source == "unmeasured"
    assert lookup_paired_sd_for_metric("").source == "unmeasured"


def test_committed_expectations_carry_per_metric_sd_slot() -> None:
    payload = json.loads(
        DEFAULT_SCREENING_EXPECTATIONS_PATH.read_text(encoding="utf-8")
    )
    assert isinstance(payload.get(OBSERVED_PAIRED_SD_BY_METRIC_KEY), dict)
    # Any recorded value must parse and be non-negative (never a borrowed copy
    # of another metric's SD is something a reader cannot verify; parse only).
    for key, value in payload[OBSERVED_PAIRED_SD_BY_METRIC_KEY].items():
        hit = lookup_paired_sd_for_metric(
            key, expectations_path=DEFAULT_SCREENING_EXPECTATIONS_PATH
        )
        assert hit.source == "metric_expectations", key
        assert Fraction(hit.observed_sd) >= 0


def test_report_envelope() -> None:
    report = compute_screening_sample_size(_obs())
    assert report.schema_version == SCREENING_SAMPLE_SIZE_SCHEMA
    assert report.promotion_authority is False
    digest = report.certificate_sha256()
    assert len(digest) == 64
    payload = report.model_dump(mode="json")
    assert payload["alpha"] == "1/20"
