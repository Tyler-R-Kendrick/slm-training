#!/usr/bin/env python3
"""RSP-002 (SLM-489): run EXP-SR-6 neural-guided bounded search fixture.

    python -m scripts.run_rsp002_bounded_search --mode plan-only
    python -m scripts.run_rsp002_bounded_search --mode fixture \\
        --out-dir outputs/runs/rsp002_bounded_search \\
        --docs-out docs/design/iter-slm489-rsp-002-bounded-search-20260810.json

``--mode plan-only`` previews the ``exp-sr-6`` catalogue identity and matched
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
from slm_training.harnesses.experiments.rsp002_bounded_search import (
    Rsp002CampaignV1,
    plan_only_preview,
    run_campaign,
)

_VERSION_COMPONENTS = (
    "harness.experiments",
    "harness.experiments.slm489_rsp002_bounded_search",
)


def _write_markdown(docs_json: Path, payload: dict) -> Path:
    md_path = docs_json.with_suffix(".md")
    falsifier = payload.get("falsifier", {})
    rates = payload.get("aggregate_hit_rates", {}).get("per_arm", {})
    reach = payload.get("reachability", {})
    invariants = payload.get("invariants", {})
    lines = [
        "# SLM-489 (RSP-002): Neural-guided bounded search + infilling order (EXP-SR-6)",
        "",
        "**Claim class:** `fixture` only (catalogue `exp-sr-6` claim_class=`screening`; "
        "execution is fixture — no promotion)",
        "",
        f"**Catalogue:** `{payload.get('catalogue_id', 'exp-sr-6')}`",
        "",
        f"**Primary metric (`bounded_search_hit_rate`, neural_prior arm):** "
        f"{payload.get('bounded_search_hit_rate')}",
        "",
        f"**Control (left_to_right):** "
        f"{payload.get('aggregate_hit_rates', {}).get('control_left_to_right')}",
        "",
        f"**Delta vs left_to_right:** "
        f"{payload.get('aggregate_hit_rates', {}).get('delta_vs_left_to_right')} "
        f"(minimum_effect=`{payload.get('minimum_effect')}`)",
        "",
        f"**Reachability (generous bounds):** `{reach.get('reachable')}`",
        "",
        f"**Recommendation:** `{payload.get('recommendation')}`",
        "",
        "## Acceptance snapshot",
        "",
        "| Check | Value |",
        "| --- | --- |",
        f"| falsifier_holds | {falsifier.get('falsifier_holds')} |",
        f"| clears_left_to_right_control | {falsifier.get('clears_left_to_right_control')} |",
        f"| clears_minimum_effect | {falsifier.get('clears_minimum_effect')} |",
        f"| singleton_bypass | {invariants.get('singleton_bypass')} |",
        f"| rankers_permute_only | {invariants.get('rankers_permute_only')} |",
        f"| energy_ranker_status | {payload.get('energy_ranker_status')} |",
        f"| promotion | {payload.get('promotion')} |",
        "",
        "## Arms (`bounded_search_hit_rate`)",
        "",
        "| Arm | hit rate |",
        "| --- | ---: |",
    ]
    for arm, rate in sorted(rates.items()):
        lines.append(f"| {arm} | {rate} |")
    lines.extend(
        [
            "",
            "## Falsifier notes",
            "",
            f"- {falsifier.get('note', '')}",
            "",
            "## Scope",
            "",
            payload.get("scope_disclaimer", ""),
            "",
            "Command: `python -m scripts.run_rsp002_bounded_search --mode fixture`",
            "",
            f"Full detail: `{docs_json.as_posix()}`.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan-only", "fixture"), default="plan-only")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/runs/rsp002_bounded_search"),
    )
    parser.add_argument(
        "--docs-out",
        type=Path,
        default=Path(
            "docs/design/iter-slm489-rsp-002-bounded-search-20260810.json"
        ),
    )
    parser.add_argument("--max-decisions", type=int, default=4)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Matched seeds (default: 0 1 2)",
    )
    args = parser.parse_args(argv)

    if args.mode == "plan-only":
        payload = plan_only_preview()
        print(json.dumps(payload, indent=2))
        return 0

    campaign = Rsp002CampaignV1(
        max_decisions=args.max_decisions,
        seeds=tuple(args.seeds),
    )
    result = run_campaign(campaign, root=args.out_dir)
    payload = {
        "kind": "rsp002_bounded_search_fixture/v1",
        "claim_class": result["claim_class"],
        "catalogue_claim_class": result["catalogue_claim_class"],
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
                "scripts.run_rsp002_bounded_search",
                "--mode",
                "fixture",
                "--max-decisions",
                str(args.max_decisions),
                "--seeds",
                *[str(s) for s in args.seeds],
            ],
            "docs_path": str(args.docs_out),
            "device": "cpu",
            "honesty_mode": "bounded_search_fixture",
            "matrix_set": "slm489_rsp002_bounded_search",
        },
        "catalogue_id": result["catalogue_id"],
        "catalogue_manifest_sha256": result["catalogue_manifest_sha256"],
        "manifest_sha256": result["manifest_sha256"],
        "max_decisions": result["max_decisions"],
        "seeds": result["seeds"],
        "energy_ranker_status": result["energy_ranker_status"],
        "bounded_search_hit_rate": result["bounded_search_hit_rate"],
        "aggregate_hit_rates": result["aggregate_hit_rates"],
        "minimum_effect": result["minimum_effect"],
        "recommendation": result["recommendation"],
        "reachability": result["reachability"],
        "openui_completion_forest_reachability": result[
            "openui_completion_forest_reachability"
        ],
        "falsifier": result["falsifier"],
        "invariants": result["invariants"],
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
        f"bounded_search_hit_rate={payload['bounded_search_hit_rate']} "
        f"reachable={payload['reachability']['reachable']} "
        f"recommendation={payload['recommendation']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
