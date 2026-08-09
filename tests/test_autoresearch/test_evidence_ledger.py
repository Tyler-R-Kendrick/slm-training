"""Deterministic evidence-ledger, posterior-selection, and power-gate tests."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

from slm_training.autoresearch import evidence_ledger as ev


def test_sign_test_min_p_exact() -> None:
    assert ev.sign_test_min_two_sided_p(3) == Fraction(1, 4)
    assert ev.sign_test_min_two_sided_p(6) == Fraction(1, 32)
    assert ev.sign_test_min_two_sided_p(0) == Fraction(1)


def test_sign_test_feasibility_floor() -> None:
    alpha = Fraction(1, 20)
    assert not ev.sign_test_feasible(3, alpha)
    assert ev.sign_test_feasible(6, alpha)
    assert ev.required_sign_test_n(alpha) == 6


def test_power_scaled_null_seeds_matches_current_policy() -> None:
    # Two null cycles of n=3 pool to 6 paired documents — exactly the sign-test
    # floor at alpha=1/20. The committed policy values are power-consistent.
    assert ev.power_scaled_null_seeds(2, 3, Fraction(1, 20)) == 2
    # A single-document cycle needs six independent nulls.
    assert ev.power_scaled_null_seeds(2, 1, Fraction(1, 20)) == 6
    # The gate never lowers the configured floor.
    assert ev.power_scaled_null_seeds(4, 20, Fraction(1, 20)) == 4


def test_parse_alpha_exact() -> None:
    assert ev.parse_alpha("1/20") == Fraction(1, 20)
    assert ev.parse_alpha("0.05") == Fraction(1, 20)
    assert ev.parse_alpha(None) == Fraction(1, 20)


def test_power_feasibility_report_typed() -> None:
    report = ev.power_feasibility_report(3, Fraction(1, 20))
    assert report["schema"] == "power_feasibility/v1"
    assert report["decisive"] is False
    assert report["required_n"] == 6


def test_slug_from_candidate_token() -> None:
    token = "c20260808-continuous-openui-1234abcd-c7-binder-topology"
    assert ev.slug_from_candidate_token(token) == "binder-topology"
    assert ev.slug_from_candidate_token("c20260808-loop-c7-control") is None
    assert ev.slug_from_candidate_token("no-cycle-marker") is None


def _cycle_record(slug: str, *, positive: bool, delta: float, cycle: int) -> dict:
    return {
        "schema": "continuous_cycle_results/v1",
        "campaign_id": f"camp-{cycle}",
        "loop_id": "loop-a",
        "cycle_index": cycle,
        "cycle_role": "screening",
        "cycle_intent": "screening",
        "primary_metric": "smoke.structural_similarity",
        "candidate_metrics": {"smoke.structural_similarity": 0.30 + delta},
        "control_metrics": {"smoke.structural_similarity": 0.30},
        "positive": positive,
        "measurement_complete": True,
        "reasons": [f"primary_metric_null_or_worse:c20260808-loop-a-ffff-c{cycle}-{slug}"],
        "evidence_class": "fixture",
        "honesty": "fixture_screening_only_not_ship",
        "stack_layer": None,
        "auto": True,
    }


def test_build_ledger_mines_cycle_records(tmp_path: Path) -> None:
    design = tmp_path / "design"
    design.mkdir()
    for cycle, delta in ((1, -0.05), (2, 0.02), (3, -0.01)):
        record = _cycle_record("bounds", positive=delta > 0, delta=delta, cycle=cycle)
        (design / f"c{cycle}-results.json").write_text(json.dumps(record))
    payload = ev.build_ledger(design)
    assert payload["schema"] == ev.LEDGER_SCHEMA
    arm = payload["arms"]["bounds"]
    assert arm["n_obs"] == 3
    assert arm["n_complete"] == 3
    assert arm["n_positive"] == 1
    assert arm["n_null"] == 2
    assert abs(arm["mean_delta"] - (-0.05 + 0.02 - 0.01) / 3) < 1e-12


def test_build_ledger_dedupes_and_is_deterministic(tmp_path: Path) -> None:
    design = tmp_path / "design"
    design.mkdir()
    record = _cycle_record("canvas", positive=False, delta=-0.02, cycle=5)
    (design / "a.json").write_text(json.dumps(record))
    (design / "b.json").write_text(json.dumps(record))  # duplicate identity
    # Campaign-less duplicates fold once via the content-digest fallback.
    orphan = dict(record)
    orphan["campaign_id"] = ""
    (design / "c.json").write_text(json.dumps(orphan))
    (design / "d.json").write_text(json.dumps(orphan))
    first = ev.build_ledger(design)
    second = ev.build_ledger(design)
    assert first == second
    assert first["arms"]["canvas"]["n_obs"] == 2
    # generated_from never embeds a machine-local absolute path.
    assert not first["generated_from"].startswith("/")


def test_measured_record_extraction() -> None:
    payload = {
        "schema_version": "autotrain_measured_result/v1",
        "campaign_id": "camp-x",
        "cycle_index": 9,
        "seed": 100009,
        "primary_metric": "smoke.structural_similarity",
        "version_stamp": {
            "stamp_schema": "version_stamp/v1",
            "components": {"evals.scoring": "v23", "gates.ship": "openui_ship_gates_v6"},
        },
        "arms": [
            {
                "experiment_id": "c20260803-loop-ffff-c9-control",
                "smoke": {"structural_similarity": 0.30},
            },
            {
                "experiment_id": "c20260803-loop-ffff-c9-component-plan",
                "smoke": {"structural_similarity": 0.38},
            },
        ],
    }
    observations = ev.extract_observations(payload, source="x.json")
    assert len(observations) == 1
    obs = observations[0]
    assert obs.slug == "component-plan"
    assert obs.seed == 100009
    assert obs.eval_key == "evals.scoring=v23|gates.ship=openui_ship_gates_v6"
    assert abs((obs.delta or 0.0) - 0.08) < 1e-12


def test_posterior_unseen_arm_has_prior_width() -> None:
    posterior = ev.arm_posterior(n=0.0, mean=0.0, m2=0.0, prior_scale=0.05)
    assert posterior.mean == 0.0
    assert abs(posterior.sd_of_mean - 0.05) < 1e-12


def test_posterior_nulls_sink_below_unseen() -> None:
    # An arm measured null many times must rank below an unexplored arm.
    null_arm = ev.arm_posterior(n=10.0, mean=-0.01, m2=0.001, prior_scale=0.05)
    unseen = ev.arm_posterior(n=0.0, mean=0.0, m2=0.0, prior_scale=0.05)
    assert null_arm.ucb(1.0) < unseen.ucb(1.0)


def test_rank_arms_orders_by_evidence() -> None:
    ledger = {
        "dead-arm": {"n_delta": 12, "mean_delta": -0.02, "m2_delta": 0.002},
        "weak-arm": {"n_delta": 6, "mean_delta": 0.04, "m2_delta": 0.001},
        "good-arm": {"n_delta": 6, "mean_delta": 0.08, "m2_delta": 0.001},
    }
    ranked = ev.rank_arms_by_evidence(
        ["dead-arm", "fresh-arm", "weak-arm", "good-arm"], ledger, exploration_c=1.0
    )
    # Strong measured evidence exploits; an unexplored arm outranks a weak
    # win (prior-width optimism = exploration); repeated nulls sink last.
    assert ranked[0] == "good-arm"
    assert ranked.index("fresh-arm") < ranked.index("weak-arm")
    assert ranked[-1] == "dead-arm"
    # Residual boosts stay lexicographically dominant, mirroring soft rank.
    boosted = ev.rank_arms_by_evidence(
        ["dead-arm", "fresh-arm", "good-arm"],
        ledger,
        exploration_c=1.0,
        residual_boosts={"dead-arm": 2.0},
    )
    assert boosted[0] == "dead-arm"


def test_rank_arms_deterministic_and_live_stats_fold() -> None:
    ledger = {"arm-a": {"n_delta": 4, "mean_delta": 0.0, "m2_delta": 0.0}}
    kwargs = dict(
        exploration_c=1.0,
        live_stats={"arm-a": (3.0, 0.12)},
        rotation_order=["arm-a", "arm-b"],
    )
    first = ev.rank_arms_by_evidence(["arm-a", "arm-b"], ledger, **kwargs)
    second = ev.rank_arms_by_evidence(["arm-a", "arm-b"], ledger, **kwargs)
    assert first == second
    # Positive live evidence lifts arm-a above the unseen arm-b.
    assert first[0] == "arm-a"


def test_ledger_roundtrip(tmp_path: Path) -> None:
    design = tmp_path / "design"
    design.mkdir()
    record = _cycle_record("bounds", positive=False, delta=-0.02, cycle=1)
    (design / "c1.json").write_text(json.dumps(record))
    payload = ev.build_ledger(design)
    target = tmp_path / "ledger.json"
    ev.write_ledger(payload, target)
    arms = ev.load_ledger(target)
    assert "bounds" in arms
    assert ev.load_ledger(tmp_path / "missing.json") == {}


def test_committed_ledger_loads_nonempty() -> None:
    # Guards the committed artifact against deletion or schema drift, which
    # would silently degrade selection to prior-only ranking.
    arms = ev.load_ledger()
    assert arms, "committed evidence_ledger.v1.json missing or invalid"


def test_partition_weighting_prefers_current_eval_key() -> None:
    buckets = {
        "evals.scoring=v23": {"n": 4, "sum": 0.2, "sumsq": 0.05},
        "unstamped": {"n": 2, "sum": -0.1, "sumsq": 0.02},
    }
    n_cur, _, _ = ev._weighted_bucket_stats(buckets, "evals.scoring=v23")
    n_none, _, _ = ev._weighted_bucket_stats(buckets, None)
    # Same-partition evidence counts at full weight only when the key matches.
    assert n_cur == 4.0 + 2.0 * ev.CROSS_PARTITION_WEIGHT
    assert n_none == (4.0 + 2.0) * ev.CROSS_PARTITION_WEIGHT


def test_regime_exhausted_verdict_shape() -> None:
    verdict = ev.build_regime_exhausted_verdict(
        campaign_id="camp",
        loop_id="loop",
        cycle_index=7,
        binding_constraint="quality_arm_bank_exhausted",
        closed_slugs=["b", "a", "a"],
        policy_sha256="0" * 64,
        resume_predicate="new lever family registered",
        bank_fingerprint="f" * 64,
    )
    assert verdict["schema_version"] == ev.VERDICT_SCHEMA
    assert verdict["closed_slugs"] == ["a", "b"]
    assert verdict["binding_constraint"] == "quality_arm_bank_exhausted"
    assert verdict["bank_fingerprint"] == "f" * 64
    # The builder's payload must satisfy the strict schema consumers rely on.
    from slm_training.autoresearch.schemas import RegimeExhaustedVerdictV1

    model = RegimeExhaustedVerdictV1.model_validate(verdict)
    assert model.resume_predicate == "new lever family registered"
    assert model.bank_fingerprint == "f" * 64
    # Fingerprint stays optional for verdicts persisted before the park/resume
    # predicate existed.
    legacy = ev.build_regime_exhausted_verdict(
        campaign_id="camp",
        loop_id="loop",
        cycle_index=7,
        binding_constraint="quality_arm_bank_exhausted",
        closed_slugs=[],
        policy_sha256=None,
        resume_predicate="new lever family registered",
    )
    assert RegimeExhaustedVerdictV1.model_validate(legacy).bank_fingerprint is None
