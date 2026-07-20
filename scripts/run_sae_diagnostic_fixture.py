#!/usr/bin/env python3
"""Run the SLM-136 LDI4-02 SAE decision-state diagnostic fixture matrix.

Example (plan only):
  python -m scripts.run_sae_diagnostic_fixture --mode plan-only

Example (fixture wiring check):
  python -m scripts.run_sae_diagnostic_fixture --mode fixture
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.harnesses.representations.interventions import run_fixture_matrix
from slm_training.harnesses.representations.spec import matched_sae_arms
from slm_training.versioning import build_version_stamp


def _today_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _plan_only_report(site: str) -> dict[str, Any]:
    """Return a torch-free plan-only report skeleton."""
    return {
        "matrix_set": "ldi4-02-sae-decision-state",
        "matrix_version": "ldi4-02-v1",
        "run_id": "ldi4_02_plan",
        "status": "plan_only",
        "claim_class": "wiring",
        "site": site,
        "d_in": 16,
        "n": 64,
        "best_baseline_effect": 0.0,
        "arms": [arm.to_dict() for arm in matched_sae_arms(site=site)],
        "note": "plan-only: no model or activation capture executed",
        "version_stamp": build_version_stamp("harness.representations"),
    }


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# SLM-136 / LDI4-02: SAE decision-state diagnostic fixture ({report.get('run_id')})",
        "",
        f"Matrix set: `{report.get('matrix_set')}`  ",
        f"Version: `{report.get('matrix_version')}`  ",
        f"Status: **{report.get('status')}**  ",
        f"Site: `{report.get('site')}`",
        "",
        "## What this measures",
        "",
        "A sparse-autoencoder diagnostic track compared against matched direct baselines "
        "(DiffMean, linear probe, ReFT-r1, direct adapter) on synthetic decision-state "
        "activations. The fixture is wiring-only: no real checkpoint, no activation capture, "
        "and no steering or interpretability claim.",
        "",
        "## Arms",
        "",
        "| Arm | Method | Selection | Target effect | Preservation damage | Wrong-site effect | Classification |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for arm in report.get("arms", []):
        lines.append(
            f"| {arm.get('arm_id')} | {arm.get('method')} | {arm.get('selection_data')} | "
            f"{arm.get('target_effect', '—')} | {arm.get('preservation_damage', '—')} | "
            f"{arm.get('wrong_site_effect', '—')} | {arm.get('classification', '—')} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "* ``diagnostic_only`` — the arm moves the target but is dominated by direct baselines.",
            "* ``causal_but_inferior`` — causal effect exists but is weaker than matched controls.",
            "* ``competitive`` — localized, within preservation budget, and not beaten by controls.",
            "* ``rejected`` — non-localized, damaging to preservation states, or otherwise unsafe.",
            "",
            "## Fixture caveat",
            "",
            report.get(
                "note",
                "Wiring-only evidence on synthetic activations. Real capture, training, and "
                "multi-suite evaluation require the GPU run.",
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SLM-136 LDI4-02 SAE decision-state diagnostic fixture"
    )
    parser.add_argument(
        "--mode",
        choices=("plan-only", "fixture"),
        default="plan-only",
        help=(
            "plan-only emits the manifest and report skeleton; "
            "fixture runs the synthetic S0-S7 matrix"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(f"outputs/runs/ldi4-02-sae-diagnostic-{_today_slug()}"),
    )
    parser.add_argument("--d-in", type=int, default=16)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--site", default="denoiser.block.0.residual")
    args = parser.parse_args(argv)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    design_json = Path(
        f"docs/design/iter-ldi4-02-sae-decision-state-diagnostic-{_today_slug()}.json"
    )
    design_md = Path(
        f"docs/design/iter-ldi4-02-sae-decision-state-diagnostic-{_today_slug()}.md"
    )

    if args.mode == "plan-only":
        report = _plan_only_report(args.site)
    else:
        report = run_fixture_matrix(
            d_in=args.d_in,
            n=args.n,
            seed=args.seed,
            site=args.site,
        )

    report_text = json.dumps(report, indent=2, sort_keys=True, default=str) + "\n"
    report_path = output_dir / "ldi4_02_sae_diagnostic_report.json"
    report_path.write_text(report_text, encoding="utf-8")
    markdown = _render_markdown(report)
    (output_dir / "ldi4_02_sae_diagnostic_report.md").write_text(
        markdown, encoding="utf-8"
    )

    design_json.parent.mkdir(parents=True, exist_ok=True)
    design_json.write_text(report_text, encoding="utf-8")
    design_md.write_text(markdown, encoding="utf-8")

    print(markdown)
    print(f"\nReport JSON: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
