#!/usr/bin/env python3
"""Run the SLM-143 SPV0-03 semantic-regret decomposition wiring/fixture harness.

Example:
  python -m scripts.run_semantic_regret_fixture --out outputs/runs/slm143/report.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.data.semantic_plan import PlanOracleSubstitutor, plan_factor_fingerprints
from slm_training.harnesses.experiments.semantic_regret_matrix import (
    SEMANTIC_REGRET_SCHEMA,
    SemanticRegretMatrixReport,
    build_fixture_graph,
    build_unreachable_graph,
    compute_regret_from_trace,
    make_semantic_plan_fixture,
    plan_regret_delta,
)
from slm_training.versioning import build_version_stamp

__all__ = ["main"]

_DESIGN_JSON = "docs/design/iter-spv0-03-semantic-regret-20260719.json"
_DESIGN_MD = "docs/design/iter-spv0-03-semantic-regret-20260719.md"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _today_yyyymmdd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _build_report() -> dict[str, Any]:
    """Build the fixture regret matrix report."""
    graph = build_fixture_graph()
    unreachable_graph = build_unreachable_graph()

    # Greedy policy: highest immediate value, even if pruned.
    baseline_report = compute_regret_from_trace("start", ["prune_high"], graph)
    # Oracle policy: traverse to the globally best accepted completion.
    oracle_report = compute_regret_from_trace("start", ["continue", "target"], graph)
    # Representation-regret arm: oracle best is declared unreachable.
    representation_report = compute_regret_from_trace(
        "start",
        ["accept_low"],
        unreachable_graph,
        oracle_best_value=2.0,
    )

    # Plan-factor substitution wiring using SemanticPlanV1.
    baseline_plan = make_semantic_plan_fixture(
        provenance="predicted",
        archetype_id="baseline_predicted",
        role_ids=["role_a"],
    )
    oracle_plan = make_semantic_plan_fixture(
        provenance="gold",
        archetype_id="oracle_gold",
        role_ids=["role_a", "role_b"],
    )
    substitutor = PlanOracleSubstitutor(
        plan_source="gold",
        oracle_factors=("archetype",),
        use_mode="seed",
        honesty_mode="oracle_diagnostic",
    )
    substituted_plan = substitutor.apply(baseline_plan, oracle_plan)

    plan_delta = plan_regret_delta(baseline_report, oracle_report)
    plan_delta["factor"] = "archetype"
    plan_delta["baseline_fingerprint"] = plan_factor_fingerprints(baseline_plan)["archetype"]
    plan_delta["oracle_fingerprint"] = plan_factor_fingerprints(oracle_plan)["archetype"]
    plan_delta["substituted_fingerprint"] = plan_factor_fingerprints(substituted_plan)[
        "archetype"
    ]
    plan_delta["substitution_banner"] = substitutor.contamination_banner()

    matrix_report = SemanticRegretMatrixReport(
        arms={
            "greedy": baseline_report,
            "oracle": oracle_report,
            "representation_unreachable": representation_report,
        },
        plan_deltas={"archetype": plan_delta},
        timestamp=_now(),
        version_stamp=build_version_stamp(
            "harness.experiments",
            "harness.experiments.semantic_regret",
        ),
    )

    return matrix_report.to_dict()


def _build_markdown(command: str) -> str:
    return f"""# SLM-143 / SPV0-03: Bounded completion enumeration and semantic regret decomposition

**Claim class:** wiring / fixture only
**Run date:** 2026-07-19
**Machine-readable result:** [`iter-spv0-03-semantic-regret-20260719.json`](iter-spv0-03-semantic-regret-20260719.json)

This iteration wires the SLM-143 semantic-regret decomposition harness. No
OpenUI checkpoint was evaluated, no GPU was used, and no ship-gate claim is
made.

## What landed

- `src/slm_training/harnesses/experiments/semantic_regret_matrix.py`
  - Frozen dataclasses: `BoundedCompletionState`, `CompletionSnapshot`,
    `RegretMetrics`, `RegretReport`, `SemanticRegretMatrixReport`.
  - Deterministic bounded completion enumerator
    (`enumerate_bounded_completions`).
  - Trace-based regret decomposition (`compute_regret_from_trace`):
    representation regret, candidate coverage, acceptable-action rank,
    local regret, pruning regret, global-rank regret, and plan-regret
    placeholder.
  - `plan_regret_delta` for per-factor delta reporting.
  - Adapter placeholders for compiler-choice, x22, and selector candidate-set
    regret decomposition.
- `scripts/run_semantic_regret_fixture.py`
  - Builds a deterministic toy graph with known exact regrets.
  - Computes greedy, oracle, and representation-regret arms.
  - Uses `PlanOracleSubstitutor` with two `SemanticPlanV1` instances to
    demonstrate factor-wise plan substitution for the `archetype` factor.
- Tests under `tests/test_harnesses/experiments/test_semantic_regret_matrix.py`
  and `tests/test_scripts/test_run_semantic_regret_fixture.py`.
- Registry entries: `harness.experiments` bumped and a new
  `harness.experiments.semantic_regret` v1 component.

## Fixture results

The toy graph contains:

- an accepted reachable branch (`accept_good` = 1.0, `accept_ok` = 0.6),
- a pruned high-value branch (`prune_high` = 1.5, `prune_cause="budget"`),
- a globally best accepted completion (`target` = 2.0) reached via
  `continue` -> `mid`,
- a disconnected unreachable target (`oracle_best` = 2.0) used to exercise
  representation regret.

Key decompositions are in the linked JSON. The fixture demonstrates that:

- choosing any accepted action yields zero local regret,
- a pruned action does not inflate scoring regret,
- an unreachable oracle-best target produces `UNKNOWN` representation regret,
- substituting the oracle `archetype` factor changes the plan-regret delta
  terms.

## Exact command

```bash
{command}
```

## Honest verdict

**`no_safe_direction` / wiring-only.** The harness compiles, the regret
terms are defined and computed on a toy graph, and the plan-factor
substitution wiring is exercised. The fixture is too small and too artificial
to tell whether the decomposition will generalize to real OpenUI completion
enumeration. A production claim would require:

- A real grammar/constrained-decode completion enumerator,
- A trained OpenUI model producing scored candidate sets,
- Oracle plans derived independently from gold programs,
- Held-out honest ship-gate suites, and
- An explicit audit that no hidden gold channel leaks into the regret terms.

Until then this is wiring and a reusable diagnostic harness, not a ship
result.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SLM-143 SPV0-03 semantic-regret decomposition wiring/fixture harness",
        exit_on_error=False,
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Path to write the JSON report (default: outputs/runs/semantic-regret-fixture-<YYYYMMDD>/regret_report.json)",
    )
    try:
        args = parser.parse_args(argv)
    except (argparse.ArgumentError, SystemExit):
        return 2

    report = _build_report()
    report["schema"] = SEMANTIC_REGRET_SCHEMA
    report["claim_class"] = "wiring"
    report["status"] = "fixture"

    out_path = (
        args.out
        or Path(f"outputs/runs/semantic-regret-fixture-{_today_yyyymmdd()}/regret_report.json")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_text = json.dumps(report, indent=2, sort_keys=True, default=str) + "\n"
    out_path.write_text(report_text, encoding="utf-8")

    # Mirror durable docs.
    root = Path(__file__).resolve().parents[1]
    json_path = root / _DESIGN_JSON
    md_path = root / _DESIGN_MD
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(report_text, encoding="utf-8")

    command = "python -m scripts.run_semantic_regret_fixture"
    if args.out is not None:
        command += f" --out {out_path}"
    md_path.write_text(_build_markdown(command), encoding="utf-8")

    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
