#!/usr/bin/env python3
"""Hermetic quantitative-bound checker fixture (HARN-07).

Self-contained oracle (no package import). Emits one JSON trailer.
Does not invent rates: nonextractable/unknown report no bound_value.
Theorem-derived extraction is validated in-process by
``extract_quantitative_bound``; this checker only drives runner judgments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _missing(required: list[str], present: list[str]) -> list[str]:
    have = set(present)
    return sorted({a for a in required if a not in have})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--statement-sha256", required=True)
    parser.add_argument("--meta-path", required=True)
    parser.add_argument(
        "--scenario",
        default="extractable",
        choices=("extractable", "nonextractable", "unknown", "malformed"),
    )
    parser.add_argument(
        "--mode",
        default="auto",
        choices=("auto", "ok", "nonextractable", "incomplete", "malformed", "sleep"),
    )
    parser.add_argument("--sleep-seconds", type=float, default=30.0)
    args = parser.parse_args(argv)

    mode = args.mode
    if mode == "auto":
        if args.scenario == "unknown":
            mode = "sleep"
        elif args.scenario == "malformed":
            mode = "malformed"
        elif args.scenario == "nonextractable":
            mode = "nonextractable"
        else:
            mode = "ok"

    if mode == "sleep":
        time.sleep(max(0.0, args.sleep_seconds))
        payload = {
            "status": "incomplete",
            "task_id": args.task_id,
            "statement_sha256": args.statement_sha256,
            "bound_value": None,
            "theorem_derived_bound": False,
            "empirical_timing_claimed": False,
            "detail": "timeout / incomplete; no fabricated bound",
        }
        print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        return 0

    if mode == "malformed":
        print("not-json-garbage")
        return 2

    meta_path = Path(args.meta_path)
    if not meta_path.is_file():
        payload = {
            "status": "incomplete",
            "task_id": args.task_id,
            "detail": f"meta missing at {meta_path}",
            "bound_value": None,
            "theorem_derived_bound": False,
            "empirical_timing_claimed": False,
        }
        print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        return 1

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(meta, dict):
        payload = {
            "status": "malformed",
            "task_id": args.task_id,
            "detail": "meta must be an object",
            "bound_value": None,
            "theorem_derived_bound": False,
            "empirical_timing_claimed": False,
        }
        print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        return 2

    if args.statement_sha256 != str(meta.get("qualitative_statement_sha256", "")):
        payload = {
            "status": "malformed",
            "task_id": args.task_id,
            "detail": "statement_sha256 mismatch vs meta",
            "bound_value": None,
            "theorem_derived_bound": False,
            "empirical_timing_claimed": False,
        }
        print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        return 2

    required = list(meta.get("required_assumptions") or [])
    present = list(meta.get("present_assumptions") or [])
    missing = _missing(required, present)

    if mode == "nonextractable":
        detail = "nonextractable: missing assumptions " + ",".join(
            missing if missing else ("(scenario)",)
        )
        notes = str(meta.get("missing_assumption_notes") or "")
        if notes:
            detail = f"{detail}; {notes}"
        proof = _sha(json.dumps({"task_id": args.task_id, "outcome": "nonextractable", "missing": missing}, sort_keys=True))
        payload = {
            "status": "nonextractable",
            "task_id": args.task_id,
            "statement_sha256": args.statement_sha256,
            "outcome": "nonextractable",
            "bound_ast_id": None,
            "bound_value": None,
            "theorem_derived_bound": False,
            "empirical_timing_claimed": False,
            "missing_assumptions": missing,
            "validated_via_bound_ast": False,
            "proof_sha256": proof,
            "detail": detail,
        }
        print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        return 0

    # extractable / ok
    if missing:
        proof = _sha(json.dumps({"task_id": args.task_id, "missing": missing}, sort_keys=True))
        payload = {
            "status": "nonextractable",
            "task_id": args.task_id,
            "statement_sha256": args.statement_sha256,
            "outcome": "nonextractable",
            "bound_ast_id": None,
            "bound_value": None,
            "theorem_derived_bound": False,
            "empirical_timing_claimed": False,
            "missing_assumptions": missing,
            "validated_via_bound_ast": False,
            "proof_sha256": proof,
            "detail": "nonextractable: missing assumptions " + ",".join(missing),
        }
        print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        return 0

    bound_ast_id = meta.get("bound_ast_id")
    bound_value = meta.get("expected_bound_value")
    if not bound_ast_id or bound_value is None:
        payload = {
            "status": "malformed",
            "task_id": args.task_id,
            "detail": "extractable meta missing bound_ast_id/expected_bound_value",
            "bound_value": None,
            "theorem_derived_bound": False,
            "empirical_timing_claimed": False,
        }
        print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        return 2

    machine = meta.get("machine_model") or {}
    if machine.get("wall_clock"):
        payload = {
            "status": "malformed",
            "task_id": args.task_id,
            "detail": "machine_model.wall_clock must be false",
            "bound_value": None,
            "theorem_derived_bound": False,
            "empirical_timing_claimed": False,
        }
        print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        return 2

    cert = {
        "bound_ast_id": bound_ast_id,
        "bound_value": str(bound_value),
        "size_parameters": meta.get("size_parameters") or {},
        "theorem_ref": meta.get("theorem_ref"),
    }
    proof = _sha(json.dumps(cert, sort_keys=True, separators=(",", ":")))
    payload = {
        "status": "ok",
        "task_id": args.task_id,
        "statement_sha256": args.statement_sha256,
        "outcome": "extracted",
        "bound_ast_id": bound_ast_id,
        "bound_value": str(bound_value),
        "theorem_derived_bound": True,
        "empirical_timing_claimed": False,
        "missing_assumptions": [],
        "validated_via_bound_ast": True,
        "proof_sha256": proof,
        "detail": f"extracted via {bound_ast_id}",
    }
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
