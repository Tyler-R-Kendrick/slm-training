"""EFS2-01 wiring fixture: X22 beam-width × edit-depth scaling over valid states.

Runs the 3×3 factorial grid requested by SLM-111
(``beam_width ∈ {1,4,16}`` × ``max_edit_depth ∈ {1,2,4}``) over a small set of
seeded synthetic programs using the existing ``TreeEditSpace``.  Every applied
edit is re-verified by the parser; invalid edits are counted but excluded from
the beam.

This is evidence-only wiring: no checkpoint is loaded, no learned policy/value
net is run, and no quality or ship claim is made.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.evals.tree_edit_scaling import (
    InferenceMode,
    run_scaling_grid,
)


def _safe_json(value: Any) -> Any:
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, dict):
        return {k: _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(v) for v in value]
    return value


def _seed_programs() -> list[str]:
    return [
        'root = Stack([n0], "column")\nn0 = TextContent(":content.body")',
        'root = Stack([n0, n1], "row")\n'
        'n0 = TextContent(":content.title")\n'
        'n1 = Button(":action.save")',
        'root = Card([n0], "card")\nn0 = TextContent(":content.body")',
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/runs/efs2-01-tree-edit-scaling"),
    )
    parser.add_argument("--search-steps", type=int, default=8)
    parser.add_argument("--expand-per-state", type=int, default=4)
    args = parser.parse_args()

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir: Path = args.output_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    inventory = [":content.body", ":content.title", ":content.subtitle", ":action.save"]
    grid = run_scaling_grid(
        _seed_programs(),
        inventory,
        seeds=(0, 1, 2),
        expand_per_state=args.expand_per_state,
        max_search_steps=args.search_steps,
        mode=InferenceMode.RANDOM_VALUE,
    )

    per_cell: dict[str, dict[str, Any]] = {}
    for cell in grid.cells:
        key = f"b{cell.beam_width}_d{cell.max_edit_depth}"
        totals = {
            "visited": 0,
            "invalid": 0,
            "duplicates": 0,
            "frozen": 0,
            "steps": 0,
            "runs": 0,
        }
        for result in cell.results:
            t = result.telemetry
            totals["visited"] += t.visited_states
            totals["invalid"] += t.invalid_attempts
            totals["duplicates"] += t.duplicate_prunes
            totals["frozen"] += t.final_frozen
            totals["steps"] += t.steps
            totals["runs"] += 1
        per_cell[key] = totals

    summary = {
        "run_id": run_id,
        "fixture": "efs2-01-tree-edit-scaling",
        "beam_widths": list(grid.beam_widths),
        "max_edit_depths": list(grid.max_edit_depths),
        "seeds": list(grid.seeds),
        "per_cell": per_cell,
        "version_stamp": _safe_json(grid.version_stamp),
    }

    json_path = out_dir / "tree_edit_scaling.json"
    json_path.write_text(
        json.dumps(_safe_json(grid.to_dict()), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    rows = "\n".join(
        f"| {key} | {d['runs']} | {d['visited']} | {d['invalid']} | {d['duplicates']} | {d['frozen']} | {d['steps']} |"
        for key, d in sorted(per_cell.items())
    )

    readme = f"""# EFS2-01 X22 Beam-Width × Edit-Depth Scaling Fixture

Run ID: `{run_id}`

This fixture exercises the SLM-111 3×3 factorial grid over the existing
``TreeEditSpace`` using a deterministic random value ranker.  Every visited
state is re-verified by the parser; invalid edits are counted but never enter
the live beam.

## Aggregate telemetry

| Cell | Runs | Visited | Invalid | Duplicates | Frozen | Steps |
| --- | --- | --- | --- | --- | --- | --- |
{rows}

## Artifacts

* `tree_edit_scaling.json` — full grid results with per-state evidence.
* `summary.json` — aggregate telemetry and version stamp.

## Honest caveats

Wiring-only evidence with a random ranker and synthetic seed programs.  A real
EFS2-01 run requires the trained X22 ``TreeEditDiffusionModel``, durable
checkpoints, the same frozen suites/evaluators across all 27 cells, and
binding-aware semantic metrics.
"""
    readme_path = out_dir / "README.md"
    readme_path.write_text(readme, encoding="utf-8")

    print(f"EFS2-01 fixture written to {out_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
