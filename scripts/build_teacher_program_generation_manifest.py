#!/usr/bin/env python3
"""Freeze SLM-266 provider budget and request identifiers before any spend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.harnesses.experiments.teacher_programs import (
    TeacherProgramGenerationManifestV1,
)
from slm_training.versioning import build_version_stamp


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-id", required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--request-ids", required=True, help="Comma-separated immutable request IDs")
    parser.add_argument("--protected-exclusion-manifest-hash", required=True)
    parser.add_argument("--max-dollars", type=float, required=True)
    parser.add_argument("--max-input-tokens", type=int, required=True)
    parser.add_argument("--max-output-tokens", type=int, required=True)
    parser.add_argument("--input-cost-per-1k-usd", type=float, default=None)
    parser.add_argument("--output-cost-per-1k-usd", type=float, default=None)
    parser.add_argument("--max-attempts", type=int, default=1)
    parser.add_argument("--campaign-manifest-sha256", default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = TeacherProgramGenerationManifestV1(
        manifest_id=args.manifest_id,
        provider=args.provider,
        model=args.model,
        revision=args.revision,
        request_ids=tuple(value for value in args.request_ids.split(",") if value),
        protected_exclusion_manifest_hash=args.protected_exclusion_manifest_hash,
        max_dollars=args.max_dollars,
        max_input_tokens=args.max_input_tokens,
        max_output_tokens=args.max_output_tokens,
        input_cost_per_1k_usd=args.input_cost_per_1k_usd,
        output_cost_per_1k_usd=args.output_cost_per_1k_usd,
        max_attempts=args.max_attempts,
        campaign_manifest_sha256=args.campaign_manifest_sha256,
    )
    payload = manifest.to_dict()
    payload["manifest_hash"] = manifest.manifest_hash
    payload["version_stamp"] = build_version_stamp("harness.experiments.slm266_teacher_programs")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest_hash": manifest.manifest_hash, "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
