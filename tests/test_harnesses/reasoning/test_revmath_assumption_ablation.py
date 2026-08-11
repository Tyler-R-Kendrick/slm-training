"""HARN-04 / SLM-544: assumption-ablation generation + validation tests."""

from __future__ import annotations

import json
from pathlib import Path

from slm_training.formal.judgment import JudgmentOutcome, UnknownReason
from slm_training.harnesses.reasoning.revmath.assumption_ablation import (
    MINIMALITY_CLAIM_SCOPE,
    audit_hidden_reintroduction,
    evaluate_ablation_lattice,
    generate_ablation_candidates,
    lattice_minimal_successes,
    materialize_candidate_task,
    parse_ablation_meta,
)
from slm_training.harnesses.reasoning.revmath.plugins import (
    AssumptionAblationPlugin,
    ablation_plugin_for_candidate,
    default_plugin_registry,
    load_ablation_meta_for_task,
    resolve_plugin,
)
from slm_training.harnesses.reasoning.revmath.runner import run_revmath_task
from slm_training.harnesses.reasoning.revmath.schemas import (
    RevmathTaskV1,
    parse_revmath_task,
)

REPO = Path(__file__).resolve().parents[3]
FIXTURES = (
    REPO / "src" / "slm_training" / "resources" / "revmath" / "fixtures"
)


def _load_pair(stem: str) -> tuple[RevmathTaskV1, object]:
    task = parse_revmath_task(
        json.loads((FIXTURES / f"{stem}.task.json").read_text(encoding="utf-8"))
    )
    meta = parse_ablation_meta(
        json.loads((FIXTURES / f"{stem}.meta.json").read_text(encoding="utf-8"))
    )
    return task, meta


def _run_lattice(task: RevmathTaskV1, meta: object):
    def run_candidate(baseline: RevmathTaskV1, candidate):
        cand_task = materialize_candidate_task(baseline, candidate)
        plugin = ablation_plugin_for_candidate(meta, candidate)  # type: ignore[arg-type]
        registry = default_plugin_registry()
        registry["assumption_ablation"] = plugin
        return run_revmath_task(cand_task, hermetic=True, plugin_registry=registry)

    return evaluate_ablation_lattice(task, meta, run_candidate=run_candidate)  # type: ignore[arg-type]


def test_generate_candidates_is_deterministic() -> None:
    task, meta = _load_pair("ablation_positive")
    first = generate_ablation_candidates(task, meta)  # type: ignore[arg-type]
    second = generate_ablation_candidates(task, meta)  # type: ignore[arg-type]
    assert [c.to_dict() for c in first] == [c.to_dict() for c in second]
    assert len(first) == 2  # none + WKL_redundant
    assert first[0].removed_assumption_ids == ()
    assert first[1].removed_assumption_ids == ("WKL_redundant",)
    assert all(
        c.statement_sha256 == task.proposition.statement_sha256 for c in first
    )


def test_positive_ablation_witnessed_under_weaker_env() -> None:
    task, meta = _load_pair("ablation_positive")
    report = _run_lattice(task, meta)
    assert report.minimality_claim_scope == MINIMALITY_CLAIM_SCOPE
    by_removed = {
        ",".join(r.candidate.removed_assumption_ids): r for r in report.records
    }
    assert by_removed[""].outcome == "witnessed"
    weaker = by_removed["WKL_redundant"]
    assert weaker.outcome == "witnessed"
    assert weaker.candidate.remaining_assumption_ids == ("ISigma1",)
    assert weaker.lattice_minimal_among_explored_successes is True
    assert by_removed[""].lattice_minimal_among_explored_successes is False


def test_redundant_assumptions_minimality_within_explored_lattice() -> None:
    task, meta = _load_pair("ablation_redundant")
    report = _run_lattice(task, meta)
    successes = [
        r.candidate for r in report.records if r.outcome == "witnessed"
    ]
    assert len(successes) == 4  # full powerset succeeds
    minimal = lattice_minimal_successes(successes)
    # Only the fully ablated remaining={ISigma1} is lattice-minimal.
    minimal_recs = [
        r for r in report.records if r.candidate.candidate_id in minimal
    ]
    assert len(minimal_recs) == 1
    assert set(minimal_recs[0].candidate.remaining_assumption_ids) == {"ISigma1"}
    for rec in report.records:
        if rec.outcome == "witnessed":
            assert rec.lattice_minimal_among_explored_successes is (
                rec.candidate.candidate_id in minimal
            )
        else:
            assert rec.lattice_minimal_among_explored_successes is None


def test_necessary_assumption_refuted_when_removed() -> None:
    task, meta = _load_pair("ablation_necessary")
    report = _run_lattice(task, meta)
    by_removed = {
        ",".join(r.candidate.removed_assumption_ids): r for r in report.records
    }
    assert by_removed[""].outcome == "witnessed"
    assert by_removed["ISigma1"].outcome == "refuted"
    assert by_removed["ISigma1"].lattice_minimal_among_explored_successes is None


def test_timeout_stays_unknown_never_refuted() -> None:
    task, meta = _load_pair("ablation_timeout")
    # Single candidate run under tight budget (fixture already 0.2s).
    plugin = AssumptionAblationPlugin(meta_override=meta)  # type: ignore[arg-type]
    registry = default_plugin_registry()
    registry["assumption_ablation"] = plugin
    record = run_revmath_task(task, hermetic=True, plugin_registry=registry)
    judgment = record.result.solver_judgment()
    assert judgment.outcome is JudgmentOutcome.UNKNOWN
    assert judgment.payload is UnknownReason.TIMEOUT
    assert record.result.proof is None


def test_hidden_import_detected_and_invalidated() -> None:
    task, meta = _load_pair("ablation_hidden_import")
    report = _run_lattice(task, meta)
    by_removed = {
        ",".join(r.candidate.removed_assumption_ids): r for r in report.records
    }
    assert by_removed[""].outcome == "witnessed"
    hidden = by_removed["Choice_strong"]
    assert hidden.outcome == "invalid"
    assert "lemma.hidden_choice" in hidden.hidden_reintroduction
    assert hidden.lattice_minimal_among_explored_successes is None


def test_audit_hidden_reintroduction_unit() -> None:
    hits = audit_hidden_reintroduction(
        removed_assumption_ids=("Choice_strong",),
        proof_dependency_ids=("lemma.hidden_choice", "ax.ISigma1"),
        import_edges={"lemma.hidden_choice": ("Choice_strong",)},
        strength_bearing_lemmas=("lemma.hidden_choice",),
    )
    assert hits == ("lemma.hidden_choice",)
    assert (
        audit_hidden_reintroduction(
            removed_assumption_ids=("Choice_strong",),
            proof_dependency_ids=("ax.ISigma1",),
            import_edges={"lemma.hidden_choice": ("Choice_strong",)},
            strength_bearing_lemmas=("lemma.hidden_choice",),
        )
        == ()
    )


def test_plugin_registered_and_replay_deterministic() -> None:
    task, _meta = _load_pair("ablation_positive")
    plugin = resolve_plugin(task)
    assert isinstance(plugin, AssumptionAblationPlugin)
    first = run_revmath_task(task, hermetic=True)
    second = run_revmath_task(task, hermetic=True)
    assert first.result.judgment == second.result.judgment
    assert first.replay_bundle.bundle_hash == second.replay_bundle.bundle_hash
    assert first.result.solver_judgment().outcome is JudgmentOutcome.WITNESSED
    assert first.result.task_identity_digest == task.identity_digest()


def test_meta_loader_matches_fixture_sidecars() -> None:
    task, meta = _load_pair("ablation_redundant")
    loaded = load_ablation_meta_for_task(task)
    assert loaded is not None
    assert loaded.to_dict() == meta.to_dict()  # type: ignore[union-attr]
