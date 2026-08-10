#!/usr/bin/env python3
"""RSP-006 (SLM-481): run EXP-SR-10 quality-diversity corpus generation.

    python -m scripts.run_rsp006_quality_diversity --mode plan-only
    python -m scripts.run_rsp006_quality_diversity --mode fixture \\
        --out-dir outputs/runs/rsp006_quality_diversity \\
        --docs-out docs/design/iter-slm481-rsp-006-quality-diversity-20260810.json

``--mode plan-only`` previews the ``exp-sr-10`` catalogue identity and matched
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
from slm_training.harnesses.experiments.rsp006_quality_diversity import (
    Rsp006CampaignV1,
    plan_only_preview,
    run_campaign,
)

_VERSION_COMPONENTS = (
    "harness.experiments",
    "harness.experiments.slm481_rsp006_qd",
)


def _write_markdown(docs_json: Path, payload: dict) -> Path:
    md_path = docs_json.with_suffix(".md")
    comparison = payload.get("comparison", {})
    arms = payload.get("arm_results", {})
    lines = [
        "# SLM-481 (RSP-006): Quality-diversity corpus generation (EXP-SR-10)",
        "",
        "**Claim class:** `fixture` only (catalogue `exp-sr-10`; not "
        "`promotion_candidate` / `ship_gate`)",
        "",
        f"**Catalogue:** `{payload.get('catalogue_id', 'exp-sr-10')}`",
        "",
        f"**Primary metric (`corpus_topology_coverage`):** "
        f"{payload.get('corpus_topology_coverage')}",
        "",
        f"**QD descriptor version:** `{payload.get('qd_descriptor_version')}`",
        "",
        f"**Recommendation:** `{payload.get('recommendation')}`",
        "",
        "## Arm comparison",
        "",
        "| Arm | Coverage | Validity | Accepted |",
        "| --- | --- | --- | --- |",
    ]
    for arm_id in ("random_matched", "frequency_balanced", "quality_diversity"):
        row = arms.get(arm_id, {})
        lines.append(
            f"| `{arm_id}` | {row.get('corpus_topology_coverage')} | "
            f"{row.get('validity_rate')} | {row.get('accepted')} |"
        )
    lines.extend(
        [
            "",
            "## Comparison snapshot",
            "",
            f"- qd_beats_random: `{comparison.get('qd_beats_random')}`",
            f"- kill_gate_triggered: `{payload.get('kill_gate_triggered')}`",
            f"- automatic_adoption: `{payload.get('automatic_adoption')}`",
            f"- promotion: `{payload.get('promotion')}`",
            "",
            "## Scope",
            "",
            payload.get("scope_disclaimer", ""),
            "",
            "Command: `python -m scripts.run_rsp006_quality_diversity --mode fixture`",
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
        default=Path("outputs/runs/rsp006_quality_diversity"),
    )
    parser.add_argument(
        "--docs-out",
        type=Path,
        default=Path(
            "docs/design/iter-slm481-rsp-006-quality-diversity-20260810.json"
        ),
    )
    parser.add_argument("--generation-budget", type=int, default=48)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args(argv)

    if args.mode == "plan-only":
        payload = plan_only_preview()
        print(json.dumps(payload, indent=2))
        return 0

    campaign = Rsp006CampaignV1(
        generation_budget=args.generation_budget,
        seed=args.seed,
        shards=args.shards,
    )
    result = run_campaign(campaign, root=args.out_dir)
    payload = {
        "kind": "rsp006_quality_diversity_fixture/v1",
        "claim_class": result["claim_class"],
        "promotion": False,
        "automatic_adoption": False,
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
        },
        "recipe": {
            "command": [
                "python",
                "-m",
                "scripts.run_rsp006_quality_diversity",
                "--mode",
                "fixture",
                "--generation-budget",
                str(args.generation_budget),
                "--seed",
                str(args.seed),
                "--shards",
                str(args.shards),
            ],
            "docs_path": str(args.docs_out),
            "device": "cpu",
            "honesty_mode": "quality_diversity_fixture",
            "matrix_set": "slm481_rsp006_quality_diversity",
        },
        "catalogue_id": result["catalogue_id"],
        "catalogue_manifest_sha256": result["catalogue_manifest_sha256"],
        "manifest_sha256": result["manifest_sha256"],
        "corpus_topology_coverage": result["corpus_topology_coverage"],
        "qd_descriptor_version": result["qd_descriptor_version"],
        "qd_descriptor_axes": result["qd_descriptor_axes"],
        "descriptor_grid_size": result["descriptor_grid_size"],
        "arm_results": result["arm_results"],
        "shards": result["shards"],
        "comparison": result["comparison"],
        "kill_gate_triggered": result["kill_gate_triggered"],
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
        f"corpus_topology_coverage={payload['corpus_topology_coverage']} "
        f"qd_beats_random={payload['comparison']['qd_beats_random']} "
        f"recommendation={payload['recommendation']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
