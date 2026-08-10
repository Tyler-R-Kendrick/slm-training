#!/usr/bin/env python3
"""RSP-007 (SLM-486): run EXP-SR-11 matched PySR/SRBench benchmark.

    python -m scripts.run_rsp007_pysr_srbench --mode plan-only
    python -m scripts.run_rsp007_pysr_srbench --mode fixture \\
        --out-dir outputs/runs/rsp007_pysr_srbench \\
        --docs-out docs/design/iter-slm486-rsp-007-pysr-srbench-20260810.json

``--mode plan-only`` previews the ``exp-sr-11`` catalogue identity and matched
arm surface without generating evidence. ``--mode fixture`` locks
ExperimentCampaignV1 via CampaignStore (claim_class=diagnostic) and writes
durable version-stamped results. When PySR is unavailable, the adapter arm
records ``evidence: external-blocked`` incomplete results — never benchmark
losses.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

from slm_training.harness_core.versioning import build_version_stamp
from slm_training.harnesses.experiments.rsp007_pysr_srbench import (
    CATALOGUE_ID,
    EVIDENCE_EXTERNAL_BLOCKED,
    PRIMARY_METRIC,
    Rsp007CampaignV1,
    plan_only_preview,
    run_campaign,
)

_VERSION_COMPONENTS = (
    "harness.experiments",
    "harness.experiments.slm486_rsp007_pysr",
)


def _write_markdown(docs_json: Path, payload: dict) -> Path:
    md_path = docs_json.with_suffix(".md")
    gap = payload.get("gap_analysis", {})
    pack = payload.get("pack_enumerate_matched", {})
    pysr = payload.get("pysr_adapter_matched", {})
    env = payload.get("env_probe", {})
    lines = [
        "# SLM-486 (RSP-007): Matched PySR/SRBench benchmark (EXP-SR-11)",
        "",
        "**Claim class:** `diagnostic` only (catalogue `exp-sr-11`; not "
        "`promotion_candidate` / `ship_gate`)",
        "",
        "**Evidence when blocked:** `external-blocked` — incomplete, not a loss.",
        "",
        "**No SOTA claims.** Fixture-scale wiring only.",
        "",
        f"**Catalogue:** `{payload.get('catalogue_id', CATALOGUE_ID)}`",
        "",
        f"**Primary metric (`{PRIMARY_METRIC}`):** `{payload.get(PRIMARY_METRIC)}`",
        "",
        f"**Complete:** `{payload.get('complete')}`",
        "",
        f"**Evidence:** `{payload.get('evidence')}`",
        "",
        "## Environment probe",
        "",
        f"- PySR import available: `{env.get('pysr_import_available')}`",
        f"- Live execution ready: `{env.get('live_execution_ready')}`",
        f"- Manifest fingerprint: `{env.get('manifest_fingerprint', '')[:16]}…`",
        f"- Tool version (pinned): `{env.get('tool_version')}`",
        "",
        "## Matched arms",
        "",
        "| Arm | Status | Validation loss | Evidence |",
        "| --- | --- | --- | --- |",
        f"| pack_enumerate_matched | {pack.get('status')} | "
        f"{pack.get('validation_loss')} | {pack.get('evidence')} |",
        f"| pysr_adapter_matched | {pysr.get('status')} | "
        f"{pysr.get('validation_loss')} | {pysr.get('evidence')} |",
        "",
        "## Gap analysis",
        "",
        f"- pack_validation_loss: `{gap.get('pack_validation_loss')}`",
        f"- pysr_validation_loss: `{gap.get('pysr_validation_loss')}`",
        f"- {PRIMARY_METRIC}: `{gap.get(PRIMARY_METRIC)}`",
        f"- note: {gap.get('note', '')}",
        "",
        f"- kill_gate_triggered: `{payload.get('kill_gate_triggered')}`",
        f"- promotion: `{payload.get('promotion')}`",
        "",
        "## Scope",
        "",
        payload.get("scope_disclaimer", ""),
        "",
        "Command: `python -m scripts.run_rsp007_pysr_srbench --mode fixture`",
        "",
        f"Full detail: `{docs_json.as_posix()}`.",
        "",
    ]
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
        default=Path("outputs/runs/rsp007_pysr_srbench"),
    )
    parser.add_argument(
        "--docs-out",
        type=Path,
        default=Path(
            "docs/design/iter-slm486-rsp-007-pysr-srbench-20260810.json"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    if args.mode == "plan-only":
        payload = plan_only_preview()
        print(json.dumps(payload, indent=2))
        return 0

    campaign = Rsp007CampaignV1(seed=args.seed)
    result = run_campaign(campaign, root=args.out_dir)
    payload = {
        "kind": "rsp007_pysr_srbench_fixture/v1",
        "claim_class": result["claim_class"],
        "promotion": False,
        "no_sota_claims": True,
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
        },
        "recipe": {
            "command": [
                "python",
                "-m",
                "scripts.run_rsp007_pysr_srbench",
                "--mode",
                "fixture",
                "--seed",
                str(args.seed),
            ],
            "docs_path": str(args.docs_out),
            "device": "cpu",
            "honesty_mode": "matched_pysr_srbench_fixture",
            "matrix_set": "slm486_rsp007_pysr_srbench",
        },
        "catalogue_id": result["catalogue_id"],
        "catalogue_manifest_sha256": result["catalogue_manifest_sha256"],
        "manifest_sha256": result["manifest_sha256"],
        "seed": result["seed"],
        "problem_fingerprint": result["problem_fingerprint"],
        "matched_iteration_budget": result["matched_iteration_budget"],
        "env_probe": result["env_probe"],
        "pack_enumerate_matched": result["pack_enumerate_matched"],
        "pysr_adapter_matched": result["pysr_adapter_matched"],
        "gap_analysis": result["gap_analysis"],
        PRIMARY_METRIC: result[PRIMARY_METRIC],
        "evidence": result["evidence"],
        "complete": result["complete"],
        "kill_threshold": result["kill_threshold"],
        "kill_operator": result["kill_operator"],
        "kill_gate_triggered": result["kill_gate_triggered"],
        "scope_disclaimer": result["scope_disclaimer"],
        "version_stamp": build_version_stamp(*_VERSION_COMPONENTS),
    }
    if payload["evidence"] == EVIDENCE_EXTERNAL_BLOCKED:
        payload["external_blocked_note"] = (
            "PySR/SRBench arm incomplete — not scored as a benchmark loss."
        )
    args.docs_out.parent.mkdir(parents=True, exist_ok=True)
    args.docs_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md_path = _write_markdown(args.docs_out, payload)
    print(f"wrote {args.docs_out}")
    print(f"wrote {md_path}")
    print(
        f"{PRIMARY_METRIC}={payload[PRIMARY_METRIC]} "
        f"evidence={payload['evidence']} "
        f"complete={payload['complete']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
