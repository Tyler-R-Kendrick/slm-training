#!/usr/bin/env python3
"""Solve mixed-precision allocation from a CAP3-04 sensitivity report.

Example::

    python -m scripts.allocate_mixed_precision \
        --sensitivity-report outputs/runs/cap3_04_sensitivity/sensitivity_report_*.json \
        --byte-budget 2048 \
        --out outputs/runs/cap3-04-allocation/allocation.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from slm_training.harnesses.quantization.allocation import allocate_mixed_precision
from slm_training.harnesses.quantization.sensitivity import SensitivityReport


def _markdown_report(manifest: Any) -> str:
    lines = [
        "# CAP3-04 mixed-precision allocation",
        "",
        f"*Run id:* `{manifest.run_id}`  ",
        f"*Sensitivity run:* `{manifest.sensitivity_run_id}`  ",
        f"*Budget:* {manifest.budget_bytes} bytes  ",
        f"*Objective:* `{manifest.objective}`  ",
        f"*Solver status:* `{manifest.solver_status}`",
        "",
        "## Allocation",
        "",
        "| group_id | format_id | bytes | cost | mean_regret | cvar90 | KL |",
        "| -------- | --------- | ----- | ---- | ----------- | ------ | -- |",
    ]
    for c in manifest.allocation:
        lines.append(
            f"| {c.group_id} | {c.format_id} | {c.bytes} | {c.cost:.4f} | "
            f"{c.mean_regret:.4f} | {c.cvar90_regret:.4f} | {c.kl_to_teacher:.4f} |"
        )
    lines.append("")
    lines.append(f"**Total bytes:** {manifest.total_bytes}  ")
    lines.append(f"**Total cost:** {manifest.total_cost:.4f}")
    lines.append("")
    lines.append("## Baselines")
    lines.append("")
    for name, baseline in manifest.baselines.items():
        if baseline is None:
            lines.append(f"- **{name}:** infeasible")
            continue
        total_bytes = sum(c.bytes for c in baseline)
        total_cost = sum(c.cost for c in baseline)
        lines.append(f"- **{name}:** {len(baseline)} groups, {total_bytes} bytes, cost {total_cost:.4f}")
    lines.append("")
    if manifest.notes:
        lines.append("## Notes")
        lines.append("")
        for note in manifest.notes:
            lines.append(f"- {note}")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensitivity-report", type=Path, required=True)
    parser.add_argument("--byte-budget", type=int, required=True)
    parser.add_argument("--tail-loss-max", type=float, default=None)
    parser.add_argument(
        "--objective",
        type=str,
        default="mean_regret",
        choices=["mean_regret", "kl_to_teacher", "weighted"],
    )
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    raw = json.loads(Path(args.sensitivity_report).read_text(encoding="utf-8"))
    report = SensitivityReport(
        version=raw["version"],
        run_id=raw["run_id"],
        timestamp=raw["timestamp"],
        checkpoint_id=raw["checkpoint_id"],
        calibration_manifest_sha=raw["calibration_manifest_sha"],
        grouping_policy_version=raw["grouping_policy_version"],
        formats=tuple(raw["formats"]),
        sample_count=raw["sample_count"],
        gradient_proxies=raw.get("gradient_proxies", {}),
        points=[],
    )
    # Points are not needed for allocation in this fixture path; allocation uses
    # the report's point list directly. Rebuild it from raw below.
    from slm_training.harnesses.quantization.sensitivity import GroupFormatPoint

    report.points = [GroupFormatPoint(**p) for p in raw["points"]]

    manifest = allocate_mixed_precision(
        report,
        args.byte_budget,
        objective=args.objective,
        tail_max=args.tail_loss_max,
        random_seed=args.random_seed,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json_path = out.with_suffix(".json") if out.suffix != ".json" else out
    json_path.write_text(manifest.to_json(indent=2), encoding="utf-8")
    md_path = json_path.with_suffix(".md")
    md_path.write_text(_markdown_report(manifest), encoding="utf-8")

    print(f"Wrote {json_path} and {md_path}")
    print(f"Solver status: {manifest.solver_status}")
    print(f"Total bytes: {manifest.total_bytes} / {manifest.budget_bytes}")
    print(f"Total cost: {manifest.total_cost:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
