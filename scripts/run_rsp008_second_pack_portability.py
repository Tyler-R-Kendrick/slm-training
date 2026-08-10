#!/usr/bin/env python3
"""RSP-008 (SLM-487): run EXP-SR-12 second-pack portability certification.

    python -m scripts.run_rsp008_second_pack_portability --mode plan-only
    python -m scripts.run_rsp008_second_pack_portability --mode diagnostic \\
        --out-dir outputs/runs/rsp008_second_pack_portability \\
        --docs-out docs/design/iter-slm487-rsp-008-portability-20260810.json

``--mode plan-only`` previews the SIE-001 ``exp-sr-12`` catalogue identity and
the matched OpenUI/SR arm surface without generating evidence.
``--mode diagnostic`` locks ExperimentCampaignV1 via CampaignStore
(claim_class=diagnostic) and writes durable version-stamped results under
``docs/design/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.harness_core.versioning import build_version_stamp
from slm_training.harnesses.experiments.rsp008_second_pack_portability import (
    Rsp008CampaignV1,
    plan_only_preview,
    run_campaign,
)

_VERSION_COMPONENTS = (
    "harness.experiments",
    "harness.experiments.slm487_rsp008_portability",
)


def _write_markdown(docs_json: Path, payload: dict) -> Path:
    md_path = docs_json.with_suffix(".md")
    analysis = payload.get("analysis", {})
    checks = analysis.get("checks", {})
    lines = [
        "# SLM-487 (RSP-008): OpenUI + symbolic-regression second-pack "
        "portability (EXP-SR-12)",
        "",
        "**Claim class:** `diagnostic` only (catalogue `exp-sr-12`; not "
        "`promotion_candidate` / `ship_gate`)",
        "",
        f"**Catalogue:** `{payload.get('catalogue_id', 'exp-sr-12')}`",
        "",
        f"**Primary metric (`second_pack_portability_parity_rate`):** "
        f"{payload.get('second_pack_portability_parity_rate')}",
        "",
        f"**Certified:** `{payload.get('certified')}`",
        "",
        "## Acceptance snapshot",
        "",
        "| Check | Value |",
        "| --- | --- |",
        f"| checks_passed / total | {analysis.get('checks_passed')} / "
        f"{analysis.get('checks_total')} |",
        f"| falsifier_holds | {analysis.get('falsifier_holds')} |",
        f"| kill_gate_triggered | {payload.get('kill_gate_triggered')} |",
        f"| rate_stable_across_seeds | {payload.get('rate_stable_across_seeds')} |",
        f"| seeds | {payload.get('seeds')} |",
        f"| promotion | {payload.get('promotion')} |",
        f"| openui_default_isolation | "
        f"{payload.get('symbolic_regression_candidate', {}).get('default_isolation', {}).get('openui_is_default')} |",
        "",
        "## Seam checks",
        "",
        "| Seam | Pass |",
        "| --- | --- |",
    ]
    for check_id, ok in checks.items():
        lines.append(f"| `{check_id}` | {ok} |")
    lines.extend(
        [
            "",
            "## Honest forks / unsupported hooks",
            "",
            f"- fork_ids: `{analysis.get('fork_ids', [])}`",
            f"- falsifier_note: {analysis.get('falsifier_note', '')}",
            "",
            "Unsupported hooks (from SRP-010 inventory):",
            "",
        ]
    )
    for hook in analysis.get("unsupported_hooks", []):
        lines.append(
            f"- `{hook.get('hook_id')}`: {hook.get('status')} — {hook.get('detail')}"
        )
    lines.extend(
        [
            "",
            "## Scope",
            "",
            payload.get("scope_disclaimer", ""),
            "",
            "Command: `python -m scripts.run_rsp008_second_pack_portability "
            "--mode diagnostic`",
            "",
            f"Full detail: `{docs_json.as_posix()}`.",
            "",
            "Precursor fixture: "
            "[iter-slm474-srp-010-portability-20260810.md]"
            "(iter-slm474-srp-010-portability-20260810.md).",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("plan-only", "diagnostic"), default="plan-only"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/runs/rsp008_second_pack_portability"),
    )
    parser.add_argument(
        "--docs-out",
        type=Path,
        default=Path(
            "docs/design/iter-slm487-rsp-008-portability-20260810.json"
        ),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Matched seeds (default: catalogue seeds 0 1 2)",
    )
    args = parser.parse_args(argv)

    if args.mode == "plan-only":
        payload = plan_only_preview()
        print(json.dumps(payload, indent=2))
        return 0

    campaign = Rsp008CampaignV1(seeds=tuple(args.seeds))
    result = run_campaign(campaign, root=args.out_dir)
    payload = {
        "kind": "rsp008_second_pack_portability_diagnostic/v1",
        "claim_class": result["claim_class"],
        "promotion": False,
        "recipe": {
            "command": [
                "python",
                "-m",
                "scripts.run_rsp008_second_pack_portability",
                "--mode",
                "diagnostic",
                "--seeds",
                *[str(s) for s in args.seeds],
            ],
            "docs_path": str(args.docs_out),
            "device": "cpu",
            "honesty_mode": "second_pack_portability_diagnostic",
            "matrix_set": "slm487_rsp008_second_pack_portability",
        },
        "catalogue_id": result["catalogue_id"],
        "catalogue_manifest_sha256": result["catalogue_manifest_sha256"],
        "manifest_sha256": result["manifest_sha256"],
        "seeds": result["seeds"],
        "second_pack_portability_parity_rate": result[
            "second_pack_portability_parity_rate"
        ],
        "rate_stable_across_seeds": result["rate_stable_across_seeds"],
        "kill_threshold": result["kill_threshold"],
        "kill_operator": result["kill_operator"],
        "kill_gate_triggered": result["kill_gate_triggered"],
        "certified": result["certified"],
        "analysis": result["analysis"],
        "openui_control": result["openui_control"],
        "symbolic_regression_candidate": result["symbolic_regression_candidate"],
        "seed_runs": result["seed_runs"],
        "scope_disclaimer": result["scope_disclaimer"],
        "version_stamp": build_version_stamp(*_VERSION_COMPONENTS),
    }
    args.docs_out.parent.mkdir(parents=True, exist_ok=True)
    args.docs_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md_path = _write_markdown(args.docs_out, payload)
    print(f"wrote {args.docs_out}")
    print(f"wrote {md_path}")
    print(
        f"second_pack_portability_parity_rate="
        f"{payload['second_pack_portability_parity_rate']} "
        f"certified={payload['certified']} "
        f"falsifier_holds={payload['analysis']['falsifier_holds']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
