"""Bounded role-slot-probe evidence report for AP-025 (SLM-318).

Demonstrates the swap/dropout/truncate role-probe orchestration over a fixed
role-factorized AbstractPlanV1 fixture with a **deterministic mock scorer** --
no model runs, no training, no held-out prompt content is copied. It emits
the per-(role, intervention) causal-effect matrix from
:mod:`slm_training.dsl.role_slot_probe`. The ship-grade run swaps the mock
scorer for a real meaning-v2 / binder-reference-F1 measurement behind the
same ``SemanticScorer`` protocol, over paired locked examples -- that is
AP-022's (SLM-313) job, not this script's.

    python scripts/run_role_slot_probe.py --out docs/design/ap-025-role-slot-probe-report-20260726.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from slm_training.dsl.abstract_plan import AbstractPlanV1, RoleSpanV1
from slm_training.dsl.role_slot_probe import (
    RoleProbeConfig,
    causal_effect_matrix,
    run_role_probes,
)

_ROLES = ("intent", "inventory", "cardinality", "topology", "bindings", "style")


def _fixture_plan() -> AbstractPlanV1:
    # 6 named ranges covering all 12 slots of a small fixture codebook --
    # illustrative sizing only, not a claim about the eventual trained arm's
    # per-role capacity.
    spans = []
    start = 0
    for role in _ROLES:
        spans.append(RoleSpanV1(role=role, start=start, length=2, max_length=1))
        start += 2
    return AbstractPlanV1(
        slot_count=12, max_slot_count=12, rounds=6, role_spans=tuple(spans)
    )


class _MockScorer:
    """Illustrative only: score rewards spreading distinct slots across
    rounds, so dropout/truncate (which collapse a role to one filler slot)
    and cross-example swap (which imports another example's slot choices)
    move the score in a legible, reproducible direction."""

    def score(self, plan_tokens: tuple[int, ...]) -> float:
        return len(set(plan_tokens)) / len(plan_tokens)


def build_report() -> dict[str, Any]:
    plan = _fixture_plan()
    pairs = [
        ((0, 1, 3, 5, 7, 9), (3, 2, 5, 4, 9, 8)),
        ((1, 0, 2, 4, 6, 8), (0, 3, 4, 5, 7, 9)),
    ]
    results = run_role_probes(plan, pairs, _MockScorer(), config=RoleProbeConfig())
    matrix = causal_effect_matrix(results)
    return {
        "note": (
            "Mock scorer; no model rollout, no verifier run, no training. A "
            "nonzero effect here demonstrates the probe wiring, not that any "
            "role carries functional semantic information -- that claim needs "
            "AP-022's (SLM-313) paired locked-example, confidence-interval "
            "evidence over a real meaning-v2/binder-reference-F1 scorer."
        ),
        "plan": plan.to_dict(),
        "roles": list(plan.role_names),
        "pair_count": len(pairs),
        "causal_effect_matrix": matrix,
        "results": [
            {
                "role": r.role,
                "intervention": r.intervention,
                "baseline_score": r.baseline_score,
                "intervened_score": r.intervened_score,
                "delta": r.delta,
            }
            for r in results
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    report = build_report()
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
