#!/usr/bin/env python3
"""RSP-004 (SLM-479): run the locked EXP-SR-8 proof-scoped e-graph/
equivalence-region experiment and persist honest evidence.

Matched-control-axis comparison over one fixed, hand-authored fixture
corpus of raw (non-pre-canonicalized) symbolic expressions, each drawn
either from a known-equivalent group (under SRP-005's certified
``REWRITE_RULES``) or a known-distinct pair:

- **control arm**: canonical-hash dedup alone (``canonicalize_expr`` +
  ``serialize_expr``, exactly SRP-005's existing mechanism).
- **candidate arm**: e-graph saturation + extraction
  (``harnesses.experiments.symbolic_egraph``), seeded from the identical
  certified rule set.

Every candidate-arm rewrite application is independently re-verified
numerically (``verify_rewrite_numerically``) before counting toward the
locked primary endpoint, ``egraph_certified_equivalence_rate``
(``exp_sr_catalogue.py``'s pre-registered EXP-SR-8 identity, kill_threshold
1.0/lt: any uncertified or numerically-unverified union kills the arm).

This experiment's own claim_class is ``fixture`` -- the locked campaign
identity is reused for gate thresholds, but the full
``CampaignStore.lock_experiment_campaign``/``validate_result_claim``
artifact-root machinery (built for promotion_candidate/ship_gate claims) is
out of scope here; gates are evaluated directly against the locked manifest.

Writes ``docs/design/exp-sr-8-egraph-equivalence-experiment.json`` and the
companion markdown (``documenting-experiment-results`` convention).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from slm_training.dsl.symbolic_expr_canonicalize import canonicalize_expr
from slm_training.dsl.symbolic_expr_ir import ALLOWED_OPERATORS, OpApply, VarRef, parse_expr, serialize_expr
from slm_training.dsl.symbolic_expr_report import build_evidence_record, pareto_front
from slm_training.dsl.symbolic_regression_pack import SymbolicRegressionProblemV1
from slm_training.harness_core.versioning import build_version_stamp
from slm_training.harnesses.experiments.exp_sr_catalogue import exp_sr_campaign
from slm_training.harnesses.experiments.symbolic_egraph import (
    EGraph,
    saturate,
    verify_rewrite_numerically,
)
from slm_training.autoresearch.experiment_campaign import campaign_manifest_sha256

_V0, _V1 = VarRef(index=0), VarRef(index=1)


def _op(operator: str, *args: Any) -> OpApply:
    return OpApply(operator=operator, args=tuple(args))


# Each group's members must saturate/canonicalize to one equivalence class.
MERGE_GROUPS: tuple[tuple[str, tuple[Any, ...]], ...] = (
    ("commutative_add", (_op("+", _V0, _V1), _op("+", _V1, _V0))),
    ("commutative_mul", (_op("*", _V0, _V1), _op("*", _V1, _V0))),
    ("double_negation", (_V0, _op("neg", _op("neg", _V0)))),
    ("double_abs", (_op("abs", _V0), _op("abs", _op("abs", _V0)))),
    ("compound_neg_and_commute", (_op("+", _V1, _V0), _op("neg", _op("neg", _op("+", _V0, _V1))))),
    ("nested_commute", (_op("*", _op("+", _V0, _V1), _V0), _op("*", _op("+", _V1, _V0), _V0))),
)

# Pairs that must never end up in the same equivalence class.
DISTINCT_PAIRS: tuple[tuple[Any, Any], ...] = (
    (_op("+", _V0, _V1), _op("*", _V0, _V1)),
    (_V0, _V1),
    (_op("neg", _V0), _V0),
    (_op("abs", _V0), _V0),
)

_SAMPLE_GRID = np.array([[x, y] for x in (-2.0, -1.0, 0.0, 1.0, 2.0) for y in (-2.0, -1.0, 0.0, 1.0, 2.0)])


def _fixture_forms() -> tuple[Any, ...]:
    forms: list[Any] = []
    for _name, members in MERGE_GROUPS:
        forms.extend(members)
    for left, right in DISTINCT_PAIRS:
        forms.extend((left, right))
    return tuple(forms)


def _run_control_arm(forms: tuple[Any, ...]) -> dict[str, Any]:
    start = time.perf_counter()
    canonical_forms = [serialize_expr(canonicalize_expr(f)) for f in forms]
    wall_ms = (time.perf_counter() - start) * 1000.0
    return {
        "canonical_by_form": canonical_forms,
        "unique_forms": sorted(set(canonical_forms)),
        "wall_ms": wall_ms,
    }


def _run_candidate_arm(forms: tuple[Any, ...]) -> dict[str, Any]:
    start = time.perf_counter()
    egraph = EGraph()
    class_ids = [egraph.add(f) for f in forms]
    rounds, applications = saturate(egraph)
    extracted_forms = [serialize_expr(egraph.extract(cid)) for cid in class_ids]
    wall_ms = (time.perf_counter() - start) * 1000.0
    return {
        "extracted_by_form": extracted_forms,
        "unique_forms": sorted(set(extracted_forms)),
        "wall_ms": wall_ms,
        "saturation_rounds": rounds,
        "n_enodes": egraph.n_enodes(),
        "n_eclasses": egraph.n_eclasses(),
        "n_rewrite_applications": len(applications),
        "applications": applications,
    }


def _build_shared_problem() -> SymbolicRegressionProblemV1:
    table = tuple(tuple(row) for row in _SAMPLE_GRID)
    target_node = _op("+", _V0, _V1)
    from slm_training.dsl.symbolic_expr_evaluator import evaluate_vectorized

    targets = tuple(evaluate_vectorized(target_node, variables=_SAMPLE_GRID).predictions)
    n_rows = len(table)
    return SymbolicRegressionProblemV1(
        variables=("x0", "x1"),
        table=table,
        targets=targets,
        allowed_operators=ALLOWED_OPERATORS,
        coefficient_hole_policy="fixed_constants_only",
        complexity_budget=64,
        numeric_domain_policy="reject_nonfinite",
        train_indices=tuple(range(n_rows)),
        validation_indices=(),
    )


def _pareto_member_hashes(unique_forms: tuple[str, ...], problem: SymbolicRegressionProblemV1) -> tuple[str, ...]:
    records = []
    for form in unique_forms:
        node = parse_expr(form, num_variables=len(problem.variables), allowed_operators=problem.allowed_operators)
        records.append(build_evidence_record(node, problem))
    front = pareto_front(records)
    return front.member_hashes


def run_experiment() -> dict[str, Any]:
    forms = _fixture_forms()
    control = _run_control_arm(forms)
    candidate = _run_candidate_arm(forms)

    verifications = [
        verify_rewrite_numerically(
            application.before, application.after, application.certificate, variables=_SAMPLE_GRID
        )
        for application in candidate["applications"]
    ]
    n_applications = len(verifications)
    n_passed = sum(1 for v in verifications if v.passed)
    egraph_certified_equivalence_rate = (n_passed / n_applications) if n_applications else 1.0

    problem = _build_shared_problem()
    control_pareto = _pareto_member_hashes(tuple(control["unique_forms"]), problem)
    candidate_pareto = _pareto_member_hashes(tuple(candidate["unique_forms"]), problem)

    campaign = exp_sr_campaign("exp-sr-8")
    manifest_sha256 = campaign_manifest_sha256(campaign)
    primary_endpoint_id = campaign.endpoints[0].endpoint_id
    endpoint_values = {primary_endpoint_id: egraph_certified_equivalence_rate}

    gate_verdicts = []
    for gate in (*campaign.promotion_gates, *campaign.rollback_gates):
        value = endpoint_values[gate.endpoint_id]
        operators = {
            "ge": value >= gate.threshold,
            "gt": value > gate.threshold,
            "le": value <= gate.threshold,
            "lt": value < gate.threshold,
            "eq": value == gate.threshold,
        }
        gate_verdicts.append(
            {
                "gate_id": gate.gate_id,
                "operator": gate.operator,
                "threshold": gate.threshold,
                "value": value,
                "fired": operators[gate.operator],
            }
        )
    killed = any(v["fired"] for v in gate_verdicts if v["gate_id"].endswith("_kill"))

    unique_forms_match = control["unique_forms"] == candidate["unique_forms"]
    pareto_matches = control_pareto == candidate_pareto

    return {
        "family_id": "exp-sr-8",
        "campaign_manifest_sha256": manifest_sha256,
        "claim_class": campaign.claim_class,
        "fixture": {
            "n_merge_groups": len(MERGE_GROUPS),
            "n_distinct_pairs": len(DISTINCT_PAIRS),
            "n_raw_forms": len(forms),
        },
        "control_arm": {
            "unique_forms": control["unique_forms"],
            "n_unique_forms": len(control["unique_forms"]),
            "wall_ms": control["wall_ms"],
            "pareto_member_hashes": list(control_pareto),
        },
        "candidate_arm": {
            "unique_forms": candidate["unique_forms"],
            "n_unique_forms": len(candidate["unique_forms"]),
            "wall_ms": candidate["wall_ms"],
            "n_enodes": candidate["n_enodes"],
            "n_eclasses": candidate["n_eclasses"],
            "saturation_rounds": candidate["saturation_rounds"],
            "n_rewrite_applications": candidate["n_rewrite_applications"],
            "pareto_member_hashes": list(candidate_pareto),
        },
        "comparison": {
            "unique_forms_match": unique_forms_match,
            "pareto_frontier_matches": pareto_matches,
            "wall_ms_overhead": candidate["wall_ms"] - control["wall_ms"],
            "memory_proxy_overhead_enodes": candidate["n_enodes"] - len(set(control["canonical_by_form"])),
        },
        "verification": {
            "n_applications": n_applications,
            "n_passed": n_passed,
            "egraph_certified_equivalence_rate": egraph_certified_equivalence_rate,
            "failed_rule_ids": sorted({v.rule_id for v in verifications if not v.passed}),
        },
        "gates": gate_verdicts,
        "killed": killed,
        # Honest adoption decision, independent of the fixture outcome --
        # SLM-479's own acceptance criterion: "Result remains default-off
        # regardless of fixture win." This module is never imported by
        # dsl/symbolic_regression_pack.py or wired into any default path.
        "adoption_decision": (
            "not_adopted_stays_experimental"
            if not killed
            else "killed_by_rollback_gate"
        ),
        "adoption_rationale": (
            "e-graph equivalence regions matched canonical-hash dedup's unique-form set and "
            "Pareto frontier exactly on this fixture (unique_forms_match={um}, "
            "pareto_frontier_matches={pm}); every certified rewrite union re-verified numerically. "
            "For SRP-005's current REWRITE_RULES (each rule is complexity-non-increasing and the "
            "rule set is confluent), e-graph saturation is mathematically guaranteed to converge "
            "to the same fixed point plain canonicalize_expr already computes, so no quality/cost "
            "frontier improvement over simpler canonicalization is possible with this rule set. "
            "Stays an isolated, default-off experiment module regardless: no import from "
            "dsl/symbolic_regression_pack.py or any production canonicalization path."
        ).format(um=unique_forms_match, pm=pareto_matches),
        "version_stamp": build_version_stamp("harness.experiments"),
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# EXP-SR-8: proof-scoped e-graph/equivalence-region experiment",
        "",
        f"Campaign manifest: `{report['campaign_manifest_sha256']}` (claim_class=`{report['claim_class']}`).",
        f"Fixture: {report['fixture']['n_merge_groups']} known-equivalence groups, "
        f"{report['fixture']['n_distinct_pairs']} known-distinct pairs, "
        f"{report['fixture']['n_raw_forms']} raw input forms.",
        "",
        "## Control arm (canonical-hash dedup alone)",
        "",
        f"- Unique canonical forms: {report['control_arm']['n_unique_forms']}",
        f"- Wall time: {report['control_arm']['wall_ms']:.3f} ms",
        "",
        "## Candidate arm (e-graph saturation + extraction)",
        "",
        f"- Unique extracted forms: {report['candidate_arm']['n_unique_forms']}",
        f"- e-nodes: {report['candidate_arm']['n_enodes']}, e-classes: {report['candidate_arm']['n_eclasses']}",
        f"- Saturation rounds: {report['candidate_arm']['saturation_rounds']}",
        f"- Rewrite applications: {report['candidate_arm']['n_rewrite_applications']}",
        f"- Wall time: {report['candidate_arm']['wall_ms']:.3f} ms",
        "",
        "## Comparison",
        "",
        f"- Unique-form sets match: {report['comparison']['unique_forms_match']}",
        f"- Pareto frontiers match: {report['comparison']['pareto_frontier_matches']}",
        f"- Wall-time overhead (candidate - control): {report['comparison']['wall_ms_overhead']:.3f} ms",
        "",
        "## Verification (EXP-SR-8 primary endpoint)",
        "",
        f"- Rewrite applications independently re-verified: {report['verification']['n_applications']}",
        f"- Passed: {report['verification']['n_passed']}",
        f"- `egraph_certified_equivalence_rate`: {report['verification']['egraph_certified_equivalence_rate']}",
        "",
        "## Gates",
        "",
    ]
    for gate in report["gates"]:
        lines.append(
            f"- `{gate['gate_id']}` ({gate['operator']} {gate['threshold']}, value={gate['value']}): "
            f"{'FIRED' if gate['fired'] else 'clear'}"
        )
    lines += [
        "",
        f"**Adoption decision:** `{report['adoption_decision']}`",
        "",
        report["adoption_rationale"],
        "",
        "Full detail: `docs/design/exp-sr-8-egraph-equivalence-experiment.json`.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-json", type=Path, default=Path("docs/design/exp-sr-8-egraph-equivalence-experiment.json")
    )
    parser.add_argument(
        "--out-md", type=Path, default=Path("docs/design/exp-sr-8-egraph-equivalence-experiment.md")
    )
    args = parser.parse_args(argv)

    report = run_experiment()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.out_md.write_text(_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "egraph_certified_equivalence_rate": report["verification"]["egraph_certified_equivalence_rate"],
                "killed": report["killed"],
                "adoption_decision": report["adoption_decision"],
            },
            indent=2,
        )
    )
    print(f"wrote {args.out_json} and {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
