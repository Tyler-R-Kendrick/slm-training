"""EFS3-01 wiring fixture: compare solver-state supervision sources.

Builds three deterministic supervision corpora from synthetic solver-state
rows: pure replay-verified gold, pure on-policy rollout, and a 50/50 DAgger-style
mix.  Reports per-source sizes, cross-split leakage rejections, and
verdict/cost preservation.

This is evidence-only wiring: no checkpoint is loaded, no model is decoded, and
no quality or ship claim is made.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.evals.solver_state_supervision import (
    SupervisionSource,
    SolverStateTrainingExampleV1,
    compare_solver_state_supervision,
)


def _safe_json(value: Any) -> Any:
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, dict):
        return {k: _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(v) for v in value]
    return value


def _make_rows(problems: int, states_per_problem: int) -> list[SolverStateTrainingExampleV1]:
    rows: list[SolverStateTrainingExampleV1] = []
    rng = {"gold": 0, "on_policy": 0}
    held_out_fraction = 0.15
    for p in range(problems):
        problem_id = f"problem-{p:04d}"
        family_id = f"family-{p % 4}"
        group_id = f"group-{p}"
        split = "test" if (p / max(problems, 1)) < held_out_fraction else "train"
        for s in range(states_per_problem):
            source = SupervisionSource.GOLD if (p + s) % 2 == 0 else SupervisionSource.ON_POLICY
            verdict = "SUPPORTED" if (p * states_per_problem + s) % 5 != 0 else "UNKNOWN"
            legal = [{"value": f"a{i}"} for i in range(4)]
            acceptable = legal[:2] if verdict == "SUPPORTED" else []
            rows.append(
                SolverStateTrainingExampleV1(
                    problem_id=problem_id,
                    state_fingerprint=f"fp-{p}-{s}-{source.value}",
                    supervision_source=source,
                    legal_actions=tuple(legal),
                    acceptable_actions=tuple(acceptable),
                    support_verdict=verdict,
                    cost_to_go=float(rng[source.value]) if verdict == "SUPPORTED" else None,
                    cost_observed=verdict == "SUPPORTED",
                    split_group_id=group_id,
                    split=split,
                    lineage_id=f"lineage-{p}",
                    program_family_id=family_id,
                    replay_certified=source is SupervisionSource.GOLD,
                )
            )
            rng[source.value] += 1
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/runs/efs3-01-solver-state-supervision"),
    )
    parser.add_argument("--problems", type=int, default=40)
    parser.add_argument("--states-per-problem", type=int, default=5)
    args = parser.parse_args()

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir: Path = args.output_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _make_rows(args.problems, args.states_per_problem)
    comparison = compare_solver_state_supervision(
        rows,
        seed=2026,
        max_rows_per_source=None,
        stamp_components=("evals.scoring",),
    )

    summary = {
        "run_id": run_id,
        "fixture": "efs3-01-solver-state-supervision",
        "synthetic_rows": len(rows),
        "held_out_group_ids": sorted(comparison.held_out_group_ids),
        "gold": comparison.gold.counts(),
        "on_policy": comparison.on_policy.counts(),
        "mixed": comparison.mixed.counts(),
        "version_stamp": _safe_json(comparison.version_stamp),
    }

    json_path = out_dir / "solver_state_supervision.json"
    json_path.write_text(
        json.dumps(_safe_json(comparison.to_dict()), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    readme = f"""# EFS3-01 Solver-State Supervision Fixture

Run ID: `{run_id}`

This fixture synthesizes solver-state training examples and builds the three
canonical supervision mixes requested by SLM-118:

* `gold` — replay-verified exact-closure states only.
* `on_policy` — solver rollout states only.
* `mixed` — DAgger-style 50/50 mix of gold and on-policy states.

## Counts

| Mix | Rows | Rejected | Sources |
| --- | --- | --- | --- |
| gold | {summary['gold']['rows']} | {summary['gold']['rejected_rows']} | {summary['gold']['source_counts']} |
| on_policy | {summary['on_policy']['rows']} | {summary['on_policy']['rejected_rows']} | {summary['on_policy']['source_counts']} |
| mixed | {summary['mixed']['rows']} | {summary['mixed']['rejected_rows']} | {summary['mixed']['source_counts']} |

Held-out split groups: {len(comparison.held_out_group_ids)}

## Artifacts

* `solver_state_supervision.json` — full comparison payload with row corpora.
* `summary.json` — headline counts and version stamp.

## Honest caveats

This is wiring-only evidence.  Rows are synthetic; no solver trace replay, no
model decode, and no ship-grade evaluation were performed.  The `UNKNOWN`
verdict rows are preserved, not relabeled.  Cross-split leakage is rejected by
`split_group_id`.
"""
    readme_path = out_dir / "README.md"
    readme_path.write_text(readme, encoding="utf-8")

    print(f"EFS3-01 fixture written to {out_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
