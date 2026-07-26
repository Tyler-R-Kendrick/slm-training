#!/usr/bin/env python3
"""Run SLM-406's shadow-only temporal failure-prediction preflight."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

from slm_training.evals.agentv import publish_agentv_evaluation
from slm_training.harnesses.experiments.operator_failure_prediction import (
    build_operator_failure_feature_rows,
)
from slm_training.versioning import build_version_stamp


ROOT = Path(__file__).resolve().parents[1]


def _portable(value: Any, output_dir: Path) -> Any:
    """Keep committed AgentV references independent of this worktree path."""
    prefix = str(output_dir.resolve())
    if isinstance(value, str) and value.startswith(prefix):
        return "agentv-dir://" + value[len(prefix) :].lstrip("/")
    if isinstance(value, list):
        return [_portable(item, output_dir) for item in value]
    if isinstance(value, dict):
        return {key: _portable(item, output_dir) for key, item in value.items()}
    return value


def _rewrite_agentv_paths(output_dir: Path) -> None:
    """Do not commit local worktree paths in generated AgentV evidence."""
    replacements = {
        str(output_dir.resolve()): "agentv-dir://",
        quote(str(output_dir.resolve()), safe=""): quote("agentv-dir://", safe=""),
        str(ROOT.resolve()): "repo://",
        quote(str(ROOT.resolve()), safe=""): quote("repo://", safe=""),
    }
    for path in (output_dir / "agentv").rglob("*"):
        if not path.is_file() or path.suffix not in {".json", ".jsonl", ".md"}:
            continue
        content = path.read_text(encoding="utf-8")
        for source, target in replacements.items():
            content = content.replace(source, target)
        path.write_text(content, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-json", action="append", type=Path, default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--issue", default="SLM-406")
    parser.add_argument("--canvas-cap", type=int, default=256)
    parser.add_argument(
        "--fixture-checkpoint-sha",
        help="explicit non-promotable identity for a fixture-only eval without a checkpoint",
    )
    parser.add_argument(
        "--preflight-reason",
        action="append",
        default=[],
        help="record an unavailable local evidence source without inventing feature rows",
    )
    args = parser.parse_args(argv)
    if args.canvas_cap <= 0:
        parser.error("--canvas-cap must be positive")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = build_version_stamp(
        "harness.model_build.eval",
        "harness.experiments.operator_failure_prediction",
    )
    rows = []
    sources = []
    for path in args.eval_json:
        payload = json.loads(path.read_text(encoding="utf-8"))
        checkpoint_sha = str(payload.get("checkpoint_sha256") or args.fixture_checkpoint_sha or "")
        if not checkpoint_sha:
            raise ValueError(f"{path} lacks checkpoint_sha256")
        built = build_operator_failure_feature_rows(
            payload.get("details") or [],
            checkpoint_sha=checkpoint_sha,
            canvas_cap=args.canvas_cap,
        )
        rows.extend(row.to_dict() for row in built)
        sources.append({"path": str(path), "checkpoint_sha256": checkpoint_sha, "rows": len(built)})
    if rows:
        verdict = "preflight_ready"
        reason = "temporal per-record DecodeStats rows are available for shadow analysis"
    elif args.preflight_reason:
        verdict = "unavailable"
        reason = "local evaluation evidence was unavailable before generation"
    else:
        verdict = "reject"
        reason = "no per-record temporal DecodeStats rows in supplied eval evidence"
    published = publish_agentv_evaluation(
        output_dir,
        name="dsh3-31-operator-failure-prediction",
        claim="shadow_only_temporal_failure_preflight_not_ship",
        version_stamp=stamp,
        cases=[
            {
                "id": "temporal-row-preflight",
                "criteria": "Shadow prediction runs only on per-record prefix-time DecodeStats evidence.",
                "pass": bool(rows),
                "result": {
                    "rows": len(rows),
                    "sources": sources,
                    "preflight_reasons": args.preflight_reason,
                },
            }
        ],
    )
    report: dict[str, Any] = {
        "schema": "operator_failure_prediction_report/v1",
        "issue": args.issue,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "shadow_only": True,
        "decision": {"verdict": verdict, "reason": reason},
        "sources": sources,
        "preflight_reasons": args.preflight_reason,
        "feature_rows": rows,
        "agentv": _portable(published, output_dir),
        "version_stamp": stamp,
    }
    _rewrite_agentv_paths(output_dir)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output_dir / "summary.md").write_text(
        f"# DSH3-31 operator failure prediction ({args.issue})\n\n"
        f"Status: `{verdict}` — {reason}. Shadow-only; no decode routing changed.\n",
        encoding="utf-8",
    )
    print(json.dumps({"report": str(output_dir / "report.json"), "rows": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
