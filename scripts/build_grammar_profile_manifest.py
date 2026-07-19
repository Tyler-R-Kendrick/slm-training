#!/usr/bin/env python3
"""Build a versioned grammar/profile manifest from decision-difficulty records.

Example::

    python -m scripts.build_grammar_profile_manifest \
        --decision-difficulties outputs/difficulties.jsonl \
        --run-id cap5-01-profiles-v1 \
        --out outputs/ladders/grammar_profile_manifest.json

Input file format: one JSON object per line, each containing a serialized
``DecisionDifficulty`` record (``schema_version: sde2-06.v1``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.dsl.analysis.arity import DecisionDifficulty
from slm_training.harnesses.experiments.grammar_profile import (
    build_grammar_profile,
    build_grammar_profile_manifest,
    validate_grammar_profile_manifest,
)


def _load_difficulties(path: Path) -> list[DecisionDifficulty]:
    difficulties: list[DecisionDifficulty] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            difficulties.append(
                DecisionDifficulty(
                    state_fingerprint=data["state_fingerprint"],
                    live_legal_action_count=data["live_legal_action_count"],
                    log2_live_legal_action_count=data["log2_live_legal_action_count"],
                    posterior_entropy_bits=data.get("posterior_entropy_bits"),
                    top1_margin=data.get("top1_margin"),
                    completion_support_size_exact=data.get("completion_support_size_exact"),
                    quotient_color=data.get("quotient_color"),
                    source_hash=data.get("source_hash"),
                    schema_version=data.get("schema_version", "sde2-06.v1"),
                )
            )
    return difficulties


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--decision-difficulties",
        type=Path,
        required=True,
        help="JSONL file of DecisionDifficulty records.",
    )
    parser.add_argument("--run-id", required=True, help="Manifest run identifier.")
    parser.add_argument("--profile-id", default="default", help="Profile identifier.")
    parser.add_argument(
        "--signature",
        default="kind=unknown",
        help="Human-readable profile signature.",
    )
    parser.add_argument(
        "--source-manifest-sha",
        default=None,
        help="Optional SHA of the source data manifest.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Destination JSON manifest path.",
    )
    parser.add_argument(
        "--note",
        default="CAP5-01 grammar/profile family manifest (wiring slice).",
        help="Free-form manifest note.",
    )
    args = parser.parse_args(argv)

    difficulties = _load_difficulties(args.decision_difficulties)
    profile = build_grammar_profile(
        difficulties,
        profile_id=args.profile_id,
        signature=args.signature,
    )
    manifest = build_grammar_profile_manifest(
        [profile],
        run_id=args.run_id,
        source_manifest_sha=args.source_manifest_sha,
        note=args.note,
    )
    errors = validate_grammar_profile_manifest(manifest)
    if errors:
        for error in errors:
            print(f"manifest validation error: {error}")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"wrote {args.out}")
    print(f"profile_count={manifest['profile_count']} manifest_hash={manifest['manifest_hash']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
