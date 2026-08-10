#!/usr/bin/env python3
"""RSP-005 (SLM-480): run EXP-SR-9 certified macro/library-learning fixture.

    python -m scripts.run_rsp005_macro_library --mode plan-only
    python -m scripts.run_rsp005_macro_library --mode fixture \\
        --out-dir outputs/runs/rsp005_macro_library \\
        --docs-out docs/design/iter-slm480-rsp-005-macro-library-20260810.json

``--mode plan-only`` previews the ``exp-sr-9`` catalogue identity and matched
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
from slm_training.harnesses.experiments.rsp005_macro_library import (
    Rsp005CampaignV1,
    plan_only_preview,
    run_campaign,
)

_VERSION_COMPONENTS = (
    "harness.experiments",
    "harness.experiments.slm480_rsp005_macros",
)


def _write_markdown(docs_json: Path, payload: dict) -> Path:
    md_path = docs_json.with_suffix(".md")
    arms = payload.get("arm_results", {})
    matched = payload.get("matched_budget", {})
    lines = [
        "# SLM-480 (RSP-005): Prospective certified macro/library learning "
        "(EXP-SR-9)",
        "",
        "**Claim class:** `fixture` only (catalogue `exp-sr-9`; not "
        "`promotion_candidate` / `ship_gate`)",
        "",
        f"**Catalogue:** `{payload.get('catalogue_id', 'exp-sr-9')}`",
        "",
        f"**Primary metric (`macro_library_size_reduction_rate`, learned_mdl arm):** "
        f"{payload.get('macro_library_size_reduction_rate')}",
        "",
        f"**Recommendation:** `{payload.get('recommendation')}`",
        "",
        "## Acceptance snapshot",
        "",
        "| Check | Value |",
        "| --- | --- |",
        f"| semantics_preserved | {payload.get('semantics_preserved')} |",
        f"| minimum_effect | {payload.get('minimum_effect')} |",
        f"| mdl_rate | {payload.get('mdl_rate')} |",
        f"| frequency_rate | {payload.get('frequency_rate')} |",
        f"| control_rate | {payload.get('control_rate')} |",
        f"| library_sources | {matched.get('library_sources')} |",
        f"| prospective_sources | {matched.get('prospective_sources')} |",
        f"| max_macros | {matched.get('max_macros')} |",
        f"| promotion | {payload.get('promotion')} |",
        "",
        "## Arms (prospective `macro_library_size_reduction_rate`)",
        "",
        "| Arm | rate | macros | pack |",
        "| --- | ---: | ---: | --- |",
    ]
    for arm_id, row in sorted(arms.items()):
        pack = row.get("pack", {})
        lines.append(
            f"| {arm_id} | {row.get('macro_library_size_reduction_rate')} | "
            f"{pack.get('n_macros', 0)} | `{pack.get('pack_sha256', '')[:12]}…` |"
        )
    lines.extend(
        [
            "",
            "## Scope",
            "",
            payload.get("scope_disclaimer", ""),
            "",
            "Command: `python -m scripts.run_rsp005_macro_library --mode fixture`",
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
        default=Path("outputs/runs/rsp005_macro_library"),
    )
    parser.add_argument(
        "--docs-out",
        type=Path,
        default=Path(
            "docs/design/iter-slm480-rsp-005-macro-library-20260810.json"
        ),
    )
    parser.add_argument("--max-macros", type=int, default=8)
    args = parser.parse_args(argv)

    if args.mode == "plan-only":
        payload = plan_only_preview()
        print(json.dumps(payload, indent=2))
        return 0

    campaign = Rsp005CampaignV1(max_macros=args.max_macros)
    result = run_campaign(campaign, root=args.out_dir)
    payload = {
        "kind": "rsp005_macro_library_fixture/v1",
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
                "scripts.run_rsp005_macro_library",
                "--mode",
                "fixture",
                "--max-macros",
                str(args.max_macros),
            ],
            "docs_path": str(args.docs_out),
            "device": "cpu",
            "honesty_mode": "macro_library_fixture",
            "matrix_set": "slm480_rsp005_macros",
        },
        "catalogue_id": result["catalogue_id"],
        "catalogue_manifest_sha256": result["catalogue_manifest_sha256"],
        "manifest_sha256": result["manifest_sha256"],
        "macro_library_size_reduction_rate": result["macro_library_size_reduction_rate"],
        "minimum_effect": result["minimum_effect"],
        "arm_results": result["arm_results"],
        "matched_budget": result["matched_budget"],
        "control_rate": result["control_rate"],
        "frequency_rate": result["frequency_rate"],
        "mdl_rate": result["mdl_rate"],
        "semantics_preserved": result["semantics_preserved"],
        "recommendation": result["recommendation"],
        "scope_disclaimer": result["scope_disclaimer"],
        "version_stamp": build_version_stamp(*_VERSION_COMPONENTS),
    }
    args.docs_out.parent.mkdir(parents=True, exist_ok=True)
    args.docs_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md_path = _write_markdown(args.docs_out, payload)
    print(f"wrote {args.docs_out}")
    print(f"wrote {md_path}")
    print(
        f"macro_library_size_reduction_rate={payload['macro_library_size_reduction_rate']} "
        f"semantics_preserved={payload['semantics_preserved']} "
        f"recommendation={payload['recommendation']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
