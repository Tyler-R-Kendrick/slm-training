"""Freeze outcome-blind train/eval domain-shift strata for SLM-265."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.dsl.schema import load_jsonl
from slm_training.harnesses.experiments.slm265_domain_shift_audit import build_coverage_gap_manifest
from slm_training.versioning import build_version_stamp


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-records", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--suites", default="smoke,held_out,adversarial,ood,rico_held")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    suites = {
        suite: load_jsonl(args.eval_dir / "suites" / suite / "records.jsonl")
        for suite in args.suites.split(",")
        if suite
    }
    payload = build_coverage_gap_manifest(load_jsonl(args.train_records), suites)
    payload["recipe"] = {"train_records": str(args.train_records), "eval_dir": str(args.eval_dir), "suites": list(suites), "honesty_mode": "outcome_blind_no_checkpoint_eval"}
    payload["version_stamp"] = build_version_stamp("harness.experiments.slm265_domain_shift_audit")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "manifest_hash": payload["manifest_hash"], "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
