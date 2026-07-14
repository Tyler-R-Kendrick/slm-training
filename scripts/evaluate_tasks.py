#!/usr/bin/env python3
"""Emit coverage-aware per-task, equivalence, and generalization scoreboards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.dsl.schema import load_jsonl
from slm_training.evals.generalization import generalization_report
from slm_training.evals.task_scoreboard import build_task_scoreboard


def _load_cases(path: Path) -> list[dict]:
    cases: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            cases.append(value)
    return cases


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-records", type=Path)
    parser.add_argument("--held-records", type=Path)
    args = parser.parse_args(argv)

    cases = _load_cases(args.cases)
    payload = {
        "run": {
            "kind": "prediction_evidence_fixture_wiring",
            "cases": str(args.cases),
            "device": "cpu",
            "steps": 0,
            "context_backend": None,
            "suite_n": len(cases),
            "honesty": "prediction_evidence_only",
        },
        "task_scoreboard": build_task_scoreboard(cases),
    }
    if bool(args.train_records) != bool(args.held_records):
        parser.error("--train-records and --held-records must be passed together")
    if args.train_records and args.held_records:
        payload["generalization"] = generalization_report(
            load_jsonl(args.train_records), load_jsonl(args.held_records)
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "cases": payload["task_scoreboard"]["n"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
