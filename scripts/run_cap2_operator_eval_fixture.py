#!/usr/bin/env python3
"""Run the frozen symbolic CAP2 operator suite and anti-cheat controls."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
import tracemalloc
from pathlib import Path
from typing import Any
from urllib.parse import quote

from slm_training.evals.agentv import publish_agentv_evaluation
from slm_training.evals.cap2_operator import (
    build_frozen_cap2_suite,
    evaluate_fixture_policies,
)
from slm_training.versioning import build_version_stamp

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    ROOT / "src/slm_training/resources/evals/cap2_operator_v1.json"
)
DEFAULT_SOURCE = (
    ROOT
    / "src/slm_training/resources/data/eval"
    / "e763_symbol_only_eval_r2_20260722/suites/held_out/records.jsonl"
)


def _portable(value: Any, output_dir: Path) -> Any:
    prefix = str(output_dir.resolve())
    if isinstance(value, str) and value.startswith(prefix):
        return "output-dir://" + value[len(prefix) :].lstrip("/")
    if isinstance(value, list):
        return [_portable(item, output_dir) for item in value]
    if isinstance(value, dict):
        return {
            key: _portable(item, output_dir) for key, item in value.items()
        }
    return value


def _rewrite_agentv_paths(output_dir: Path) -> None:
    """Remove checkout-local paths from the committed SDK result bundle."""
    raw_prefix = str(output_dir.resolve())
    encoded_prefix = quote(raw_prefix, safe="")
    for path in (output_dir / "agentv").rglob("*"):
        if not path.is_file() or path.suffix not in {".json", ".jsonl"}:
            continue
        text = path.read_text(encoding="utf-8")
        text = text.replace(raw_prefix, "output-dir://")
        text = text.replace(
            encoded_prefix,
            quote("output-dir://", safe=""),
        )
        path.write_text(text, encoding="utf-8")


def _agentv_cases(
    suite: dict[str, Any], scores: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    controls = ("unchanged", "generic_valid_ast", "constant_operator")
    return [
        {
            "id": "frozen-suite-identity",
            "criteria": "Regenerated held-out CAP2 gold matches the frozen suite hash.",
            "pass": True,
            "failures": [],
            "result": {
                "suite_hash": suite["suite_hash"],
                "case_count": len(suite["cases"]),
            },
            "metadata": {"honesty": "fixture_contract"},
        },
        {
            "id": "oracle-replay",
            "criteria": "Every replay-authoritative gold row clears all applicable CAP2 dimensions.",
            "pass": scores["oracle"]["gate_pass"],
            "failures": (
                []
                if scores["oracle"]["gate_pass"]
                else ["oracle failed frozen CAP2 contract"]
            ),
            "result": scores["oracle"],
            "metadata": {"policy": "oracle_fixture"},
        },
        *[
            {
                "id": f"anti-cheat-{name}",
                "criteria": f"The {name} degenerate policy must not pass CAP2.",
                "pass": not scores[name]["gate_pass"],
                "failures": (
                    []
                    if not scores[name]["gate_pass"]
                    else [f"{name} incorrectly passed CAP2"]
                ),
                "result": {
                    "gate_pass": scores[name]["gate_pass"],
                    "case_successes": scores[name]["case_successes"],
                    "case_count": scores[name]["case_count"],
                },
                "metadata": {"policy": name, "control": "anti_cheat"},
            }
            for name in controls
        ],
        {
            "id": "symbol-only-boundary",
            "criteria": "No NL suite row is emitted before CERT_CAP1 is available.",
            "pass": suite["nl"]["available"] is False,
            "failures": (
                []
                if suite["nl"]["available"] is False
                else ["uncertified NL suite row emitted"]
            ),
            "result": suite["nl"],
            "metadata": {"dependency": "SLM-379"},
        },
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--source-records", type=Path, default=DEFAULT_SOURCE)
    args = parser.parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = build_version_stamp(
        "evals.cap2_operator",
        "dsl.operators.contracts",
        "dsl.operators.conversation",
        "dsl.operators.merge",
        "dsl.operators.collapse",
        "dsl.operators.legal_set",
    )
    started = time.perf_counter()
    tracemalloc.start()
    with tempfile.TemporaryDirectory(prefix="cap2-operator-eval-") as temp_dir:
        suite = build_frozen_cap2_suite(
            manifest_path=args.manifest.resolve(),
            source_records_path=args.source_records.resolve(),
            work_dir=Path(temp_dir),
            version_stamp=stamp,
        )
    scores = evaluate_fixture_policies(suite)
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    elapsed = time.perf_counter() - started
    agentv = publish_agentv_evaluation(
        output_dir,
        name="cap2-operator-fixture-v1",
        claim="frozen_cap2_symbolic_fixture_contract_not_ship",
        cases=_agentv_cases(suite, scores),
    )
    _rewrite_agentv_paths(output_dir)
    report = {
        "schema": "cap2_operator_fixture_report/v1",
        "run": {
            "kind": "frozen_symbolic_operator_fixture",
            "device": "cpu",
            "steps": 0,
            "context_backend": "none",
            "matrix_set": "cap2_operator_v1",
            "suite_n": len(suite["cases"]),
            "honesty": "fixture_wiring_only",
            "max_wall_minutes": 3,
            "elapsed_seconds": elapsed,
            "peak_memory_bytes": peak_memory,
            "checkpoint": None,
            "ship_claim": False,
        },
        "suite": suite,
        "policy_scores": scores,
        "agentv": _portable(agentv, output_dir),
        "version_stamp": stamp,
    }
    (output_dir / "suite.json").write_text(
        json.dumps(suite, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "report": str(output_dir / "report.json"),
                "suite_hash": suite["suite_hash"],
                "suite_n": len(suite["cases"]),
                "oracle_pass": scores["oracle"]["gate_pass"],
                "controls_rejected": all(
                    not scores[name]["gate_pass"]
                    for name in (
                        "unchanged",
                        "generic_valid_ast",
                        "constant_operator",
                    )
                ),
                "elapsed_seconds": elapsed,
                "peak_memory_bytes": peak_memory,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
