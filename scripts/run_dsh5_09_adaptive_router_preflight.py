#!/usr/bin/env python3
"""Record SLM-417's bounded local admission disposition."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from slm_training.evals.agentv import publish_agentv_evaluation
from slm_training.harness_core.versioning import build_version_stamp
from slm_training.harnesses.experiments.operator_router import (
    ExecutionModeV1,
    ExecutionRouterInputV1,
    choose_execution_mode,
)


ROOT = Path(__file__).resolve().parents[1]


def _portable(value: object, *, output_dir: Path) -> object:
    if isinstance(value, str):
        prefix = str(output_dir.resolve())
        return "agentv-dir://" + value[len(prefix) :].lstrip("/") if value.startswith(prefix) else value
    if isinstance(value, list):
        return [_portable(item, output_dir=output_dir) for item in value]
    if isinstance(value, dict):
        return {key: _portable(item, output_dir=output_dir) for key, item in value.items()}
    return value


def _rewrite_agentv_paths(output_dir: Path) -> None:
    replacements = {
        str(output_dir.resolve()): "agentv-dir://",
        quote(str(output_dir.resolve()), safe=""): quote("agentv-dir://", safe=""),
        str(ROOT.resolve()): "repo://",
        quote(str(ROOT.resolve()), safe=""): quote("repo://", safe=""),
    }
    for path in (output_dir / "agentv").rglob("*"):
        if path.is_file() and path.suffix in {".json", ".jsonl", ".md"}:
            content = path.read_text(encoding="utf-8")
            for source, replacement in replacements.items():
                content = content.replace(source, replacement)
            path.write_text(content, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    router_input = ExecutionRouterInputV1(features={}, available_modes=frozenset())
    decision = choose_execution_mode(router_input)
    stamp = build_version_stamp("harness.experiments.operator_router")
    disposition = {
        "verdict": "unavailable",
        "reason": (
            "No locked, group-disjoint matched all-arm outcomes exist; AP-028's "
            "fixture proxy cannot admit learned-router training or a quality/work claim."
        ),
        "required_next_evidence": [
            "matched per-request outcomes for every fixed and candidate arm",
            "prefix-time-only features with oracle/outcome fields rejected",
            "frozen group-disjoint calibration and held-out evaluation",
        ],
    }
    agentv = publish_agentv_evaluation(
        output_dir,
        name="dsh5-09-adaptive-router-preflight",
        claim="bounded_local_admission_disposition_not_router_quality_claim",
        version_stamp=stamp,
        cases=[
            {
                "id": "unavailable-modes-defer",
                "criteria": "No compiler-authorized mode yields explicit DEFER without a model forward.",
                "pass": decision.mode is ExecutionModeV1.DEFER and decision.model_forwards == 0,
                "result": {"mode": decision.mode.value, "model_forwards": decision.model_forwards},
            },
            {
                "id": "no-fixture-promotion",
                "criteria": "Absent matched all-arm evidence remains unavailable, not a learned-router claim.",
                "pass": disposition["verdict"] == "unavailable",
                "result": disposition,
            },
        ],
    )
    _rewrite_agentv_paths(output_dir)
    payload = {
        "schema": "dsh5_09_adaptive_router_preflight/v1",
        "run": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "device": "cpu",
            "backend": "local_contract_preflight",
            "suite_n": 0,
            "primary_metric": "meaningful_parse",
            "ship_claim": False,
            "human_rating_gate": False,
        },
        "disposition": disposition,
        "decision": {"mode": decision.mode.value, "source": decision.source},
        "agentv": _portable(agentv, output_dir=output_dir),
        "version_stamp": stamp,
    }
    (output_dir / "report.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (output_dir / "summary.md").write_text(
        "# DSH5-09 adaptive-router admission preflight\n\n"
        "Verdict: **unavailable**. The local contract confirms unavailable modes DEFER, "
        "but no matched group-disjoint all-arm corpus exists to train or evaluate a router. "
        "This is not a quality, work, checkpoint, or ship claim.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
