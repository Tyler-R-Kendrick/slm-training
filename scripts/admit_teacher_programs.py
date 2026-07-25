#!/usr/bin/env python3
"""Admit archived SLM-266 teacher responses without calling a provider."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from slm_training.harnesses.experiments.teacher_programs import (
    AdmissionMode,
    TeacherProgramCandidate,
    admit_teacher_programs,
)
from slm_training.versioning import build_version_stamp


def _fingerprints(data: dict[str, Any]) -> dict[str, set[str]]:
    return {key: set(map(str, data.get(key, ())) ) for key in (
        "ids", "split_group_ids", "prompts", "openuis", "structures", "pairs", "design_mds"
    )}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True, help="JSON list of immutable teacher response records")
    parser.add_argument("--protected-fingerprints", type=Path, required=True)
    parser.add_argument("--mode", choices=[mode.value for mode in AdmissionMode], required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    candidates = [TeacherProgramCandidate(**row) for row in json.loads(args.raw.read_text(encoding="utf-8"))]
    protected = _fingerprints(json.loads(args.protected_fingerprints.read_text(encoding="utf-8")))
    result = admit_teacher_programs(candidates, mode=args.mode, protected_fingerprints=protected).to_dict()
    result["version_stamp"] = build_version_stamp("harness.experiments.slm266_teacher_programs")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"accepted_n": result["accepted_n"], "rejected_n": result["rejected_n"], "manifest_hash": result["manifest_hash"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
