#!/usr/bin/env python3
"""Produce or verify the deterministic SLM-207 valid-edit-flow closeout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.harnesses.experiments.valid_edit_flow_closeout import render_markdown, run_closeout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("describe", "plan", "run", "verify"), default="plan")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/runs/slm207-valid-edit-flow-closeout"))
    parser.add_argument("--design-dir", type=Path, default=Path("docs/design"))
    args = parser.parse_args(argv)
    if args.mode == "describe":
        print("Locks committed VFA evidence and fail-closes unsupported learned-flow dispositions.")
        return 0
    report = run_closeout()
    payload = report.to_dict()
    if args.mode == "plan":
        print(json.dumps({"issue": payload["issue"], "decision": payload["decision"], "inputs": payload["artifact_lock"]}, indent=2))
        return 0
    if args.mode == "verify":
        existing = json.loads((args.design_dir / "valid-edit-flow-closeout.json").read_text(encoding="utf-8"))
        for field in ("decision", "artifact_lock", "dispositions", "selected_stack"):
            if existing.get(field) != payload.get(field):
                raise ValueError(f"closeout verification failed: {field} differs from artifact lock")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    (args.output_dir / "valid-edit-flow-closeout.json").write_text(rendered, encoding="utf-8")
    (args.output_dir / "valid-edit-flow-closeout.md").write_text(render_markdown(report), encoding="utf-8")
    if args.mode == "run":
        args.design_dir.mkdir(parents=True, exist_ok=True)
        (args.design_dir / "valid-edit-flow-closeout.json").write_text(rendered, encoding="utf-8")
        (args.design_dir / "valid-edit-flow-closeout.md").write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
