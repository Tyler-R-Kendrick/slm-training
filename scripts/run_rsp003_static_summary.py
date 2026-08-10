#!/usr/bin/env python3
"""RSP-003 (SLM-485): run EXP-SR-7 static-artifact + packed semantic-summary fixture.

    python -m scripts.run_rsp003_static_summary --mode plan-only
    python -m scripts.run_rsp003_static_summary --mode fixture \\
        --out-dir outputs/runs/rsp003_static_summary \\
        --docs-out docs/design/iter-slm485-rsp-003-static-summary-20260810.json

``--mode plan-only`` previews the ``exp-sr-7`` catalogue identity and matched
arm surface without generating evidence. ``--mode fixture`` locks
ExperimentCampaignV1 via CampaignStore (claim_class=fixture) and writes durable
version-stamped results under ``docs/design/``.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

from slm_training.harness_core.versioning import build_version_stamp
from slm_training.harnesses.experiments.rsp003_static_summary import (
    Rsp003CampaignV1,
    plan_only_preview,
    run_campaign,
)

_VERSION_COMPONENTS = (
    "harness.experiments",
    "harness.experiments.slm485_rsp003_static_summary",
)


def _write_markdown(docs_json: Path, payload: dict) -> Path:
    md_path = docs_json.with_suffix(".md")
    parity = payload.get("parity_analysis", {})
    checks = parity.get("checks", {})
    cold = payload.get("cold_path_improvement", {})
    lines = [
        "# SLM-485 (RSP-003): Extended static-artifact + packed semantic-summary "
        "(EXP-SR-7)",
        "",
        "**Claim class:** `fixture` only (catalogue `exp-sr-7`; not "
        "`promotion_candidate` / `ship_gate`)",
        "",
        f"**Catalogue:** `{payload.get('catalogue_id', 'exp-sr-7')}`",
        "",
        f"**Primary metric (`static_summary_domain_parity_rate`):** "
        f"{payload.get('static_summary_domain_parity_rate')}",
        "",
        f"**Certified:** `{parity.get('certified')}`",
        "",
        f"**Recommendation:** `{payload.get('recommendation')}`",
        "",
        "## Acceptance snapshot",
        "",
        "| Check | Value |",
        "| --- | --- |",
        f"| checks_passed / total | {parity.get('checks_passed')} / "
        f"{parity.get('checks_total')} |",
        f"| kill_gate_triggered | {parity.get('kill_gate_triggered')} |",
        f"| cold_path_improved | {cold.get('cold_path_improved')} |",
        f"| best_candidate_arm | {cold.get('best_candidate_arm')} |",
        f"| best_delta_ms | {cold.get('best_delta_ms')} |",
        f"| promotion | {payload.get('promotion')} |",
        "",
        "## Parity checks",
        "",
        "| Check | Pass |",
        "| --- | --- |",
    ]
    for check_id, ok in checks.items():
        lines.append(f"| `{check_id}` | {ok} |")
    lines.extend(
        [
            "",
            "## Cold-path summary-step deltas vs base_artifact",
            "",
            f"- baseline_summary_step_p50_ms: {cold.get('baseline_summary_step_p50_ms')}",
            f"- deltas: `{cold.get('delta_vs_baseline_ms', {})}`",
            "",
            "## Scope",
            "",
            payload.get("scope_disclaimer", ""),
            "",
            "Command: `python -m scripts.run_rsp003_static_summary --mode fixture`",
            "",
            f"Full detail: `{docs_json.as_posix()}`.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("plan-only", "fixture"), default="plan-only"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/runs/rsp003_static_summary"),
    )
    parser.add_argument(
        "--docs-out",
        type=Path,
        default=Path(
            "docs/design/iter-slm485-rsp-003-static-summary-20260810.json"
        ),
    )
    parser.add_argument("--cold-trials-per-arm", type=int, default=2)
    parser.add_argument("--warm-trials", type=int, default=5)
    parser.add_argument("--timeout-s", type=float, default=20.0)
    args = parser.parse_args(argv)

    if args.mode == "plan-only":
        payload = plan_only_preview()
        print(json.dumps(payload, indent=2))
        return 0

    campaign = Rsp003CampaignV1(
        cold_trials_per_arm=args.cold_trials_per_arm,
        warm_trials=args.warm_trials,
    )
    result = run_campaign(campaign, root=args.out_dir, timeout_s=args.timeout_s)
    payload = {
        "kind": "rsp003_static_summary_fixture/v1",
        "claim_class": result["claim_class"],
        "promotion": False,
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
        },
        "recipe": {
            "command": [
                "python",
                "-m",
                "scripts.run_rsp003_static_summary",
                "--mode",
                "fixture",
                "--cold-trials-per-arm",
                str(args.cold_trials_per_arm),
                "--warm-trials",
                str(args.warm_trials),
            ],
            "docs_path": str(args.docs_out),
            "device": "cpu",
            "honesty_mode": "static_summary_fixture",
            "matrix_set": "slm485_rsp003_static_summary",
        },
        "catalogue_id": result["catalogue_id"],
        "catalogue_manifest_sha256": result["catalogue_manifest_sha256"],
        "manifest_sha256": result["manifest_sha256"],
        "static_summary_domain_parity_rate": result["static_summary_domain_parity_rate"],
        "parity_analysis": result["parity_analysis"],
        "phase_breakdown_by_arm": result["phase_breakdown_by_arm"],
        "cold_path_improvement": result["cold_path_improvement"],
        "recommendation": result["recommendation"],
        "summary": result["summary"],
        "trials": result["trials"],
        "scope_disclaimer": result["scope_disclaimer"],
        "version_stamp": build_version_stamp(*_VERSION_COMPONENTS),
    }
    args.docs_out.parent.mkdir(parents=True, exist_ok=True)
    args.docs_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md_path = _write_markdown(args.docs_out, payload)
    print(f"wrote {args.docs_out}")
    print(f"wrote {md_path}")
    print(
        f"static_summary_domain_parity_rate="
        f"{payload['static_summary_domain_parity_rate']} "
        f"certified={payload['parity_analysis']['certified']} "
        f"recommendation={payload['recommendation']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
