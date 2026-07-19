"""CAP4-03 wiring fixture: quantize local legal-action energies and compare inference.

Synthesizes small acyclic decision lattices, quantizes local energies to the
low-bit formats from CAP4-03, and compares greedy-local selection with exact
global selection.  Reports per-format path choices, tie/collision statistics,
and the gap between greedy and exact totals.

This is evidence-only wiring: no checkpoint is loaded, no learned scorer is run,
and no quality or ship claim is made.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.evals.quantized_energy_inference import (
    EnergyProblem,
    EnergyStage,
    LegalAction,
    ScoreSemantics,
    compare_quantized_energy_inference,
)


def _safe_json(value: Any) -> Any:
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, dict):
        return {k: _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(v) for v in value]
    return value


def _make_problem(problem_id: str, seed: int) -> EnergyProblem:
    rng = {
        "i": seed,
    }

    def rand() -> float:
        rng["i"] = (rng["i"] * 1103515245 + 12345) & 0x7FFFFFFF
        return (rng["i"] / 0x7FFFFFFF) * 10.0

    stages: list[EnergyStage] = []
    for s in range(4):
        actions: list[LegalAction] = []
        for a in range(3 + (s % 2)):
            energy = round(rand(), 3)
            known = (s * 7 + a) % 5 != 0
            actions.append(
                LegalAction(
                    action_id=f"p{problem_id}-s{s}-a{a}",
                    local_energy=energy if known else 0.0,
                    known=known,
                )
            )
        stages.append(EnergyStage(stage_id=f"s{s}", actions=tuple(actions)))
    return EnergyProblem(
        problem_id=problem_id,
        stages=tuple(stages),
        semantics=ScoreSemantics.ADDITIVE_EDGE,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/runs/cap4-03-quantized-energy-inference"),
    )
    parser.add_argument("--problems", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir: Path = args.output_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    problems = [
        _make_problem(f"prob{i:02d}", args.seed + i * 97)
        for i in range(args.problems)
    ]
    comparisons = [
        compare_quantized_energy_inference(p) for p in problems
    ]

    per_format: dict[str, dict[str, Any]] = {}
    for comp in comparisons:
        for fr in comp.format_results:
            fid = fr.quantizer.fmt.format_id
            entry = per_format.setdefault(fid, {
                "format_id": fid,
                "problems": 0,
                "greedy_total": 0.0,
                "exact_total": 0.0,
                "greedy_exact_agree": 0,
                "max_tie_class": 0,
                "total_path_count": 0,
            })
            entry["problems"] += 1
            entry["greedy_total"] += fr.greedy.total_quantized_energy
            entry["exact_total"] += fr.exact.total_quantized_energy
            if fr.greedy.path == fr.exact.path:
                entry["greedy_exact_agree"] += 1
            entry["max_tie_class"] = max(
                entry["max_tie_class"], fr.exact.tie_class_size
            )
            entry["total_path_count"] += fr.exact.path_count_considered

    summary = {
        "run_id": run_id,
        "fixture": "cap4-03-quantized-energy-inference",
        "problems": len(problems),
        "per_format": per_format,
        "version_stamp": _safe_json(comparisons[0].version_stamp) if comparisons else None,
    }

    json_path = out_dir / "quantized_energy_inference.json"
    json_path.write_text(
        json.dumps(_safe_json([c.to_dict() for c in comparisons]), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    rows = "\n".join(
        f"| {fid} | {d['problems']} | {d['greedy_total']:.3f} | {d['exact_total']:.3f} | "
        f"{d['greedy_exact_agree']} | {d['max_tie_class']} | {d['total_path_count']} |"
        for fid, d in sorted(per_format.items())
    )

    readme = f"""# CAP4-03 Quantized Energy Inference Fixture

Run ID: `{run_id}`

This fixture synthesizes small acyclic decision lattices and compares
**greedy-local** selection with **exact-global** selection after quantizing
local energies to several low-bit formats.

## Aggregate comparison

| Format | Problems | Greedy total | Exact total | Greedy==Exact | Max tie class | Paths enumerated |
| --- | --- | --- | --- | --- | --- | --- |
{rows}

## Artifacts

* `quantized_energy_inference.json` — per-problem format results and selections.
* `summary.json` — aggregate totals and version stamp.

## Honest caveats

Wiring-only evidence with synthetic additive energies.  Real CAP4-03 requires a
learned local energy scorer, actual completion forests, calibrated quantization,
and a bounded search controller.  The exact mode here is exhaustive enumeration
over a small acyclic lattice, not a deployment solver.
"""
    readme_path = out_dir / "README.md"
    readme_path.write_text(readme, encoding="utf-8")

    print(f"CAP4-03 fixture written to {out_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
