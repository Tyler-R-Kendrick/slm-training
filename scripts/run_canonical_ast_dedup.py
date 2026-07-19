#!/usr/bin/env python3
"""Run the SLM-130 EFS3-05 canonical AST dedup wiring/fixture harness.

Examples:
  python -m scripts.run_canonical_ast_dedup --fixture --out outputs/runs/slm130/report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from slm_training.dsl.canonicalize import canonicalize
from slm_training.harnesses.experiments.canonical_ast_dedup import (
    RepresentativePolicy,
    dedup_arms_for_pool,
    group_candidates_by_canonical_ast,
    unique_slot_truncation,
)
from slm_training.versioning import build_version_stamp

__all__ = ["main"]


_ARM_LABELS = ("A_raw_no_dedup", "B_exact_output_dedup", "C_terminal_canonical_ast", "D_unique_slot_truncation", "E_abstract_mode_spread")


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _make_fixture_candidates() -> tuple[tuple[str, str, dict[str, Any]], ...]:
    """Return a small fixed pool with intentional duplicates and variants.

    The pool contains:
      * two alpha-equivalent surface variants of the same valid program
      * a semantically distinct valid program
      * an invalid/unparseable candidate
      * an unknown-verdict candidate
    """
    # A valid simple OpenUI program.
    base = 'root = Stack([x])\nx = TextContent(":title")'
    # Alpha-equivalent: binder renamed.
    alpha = 'root = Stack([y])\ny = TextContent(":title")'
    # Semantically distinct.
    distinct = 'root = Stack([z])\nz = Card([t])\nt = TextContent(":body")'
    # Invalid.
    invalid = "this is not valid openui {"
    # Unknown verdict.
    unknown = 'root = Stack([u])\nu = TextContent(":maybe")'

    # Canonicalize all valid ones to confirm they collapse.
    canonical_base = canonicalize(base)
    canonical_alpha = canonicalize(alpha)
    canonical_distinct = canonicalize(distinct)
    canonical_unknown = canonicalize(unknown)

    candidates = (
        ("c0_base", canonical_base, {"valid": True, "contract_satisfied": True, "generator_score": 0.9, "semantic_success": True}),
        ("c1_alpha", canonical_alpha, {"valid": True, "contract_satisfied": True, "generator_score": 0.88, "semantic_success": True}),
        ("c2_distinct", canonical_distinct, {"valid": True, "contract_satisfied": False, "generator_score": 0.7, "semantic_success": False}),
        ("c3_invalid", invalid, {"valid": False, "contract_satisfied": False, "generator_score": 0.5, "semantic_success": False}),
        ("c4_unknown", canonical_unknown, {"valid": True, "contract_satisfied": False, "unknown": True, "generator_score": 0.6, "semantic_success": False}),
    )
    return candidates


def _run_fixture() -> dict[str, Any]:
    candidates = _make_fixture_candidates()
    prompt_hash = "fixture_prompt_abc123"
    arms = dedup_arms_for_pool(candidates, prompt_hash=prompt_hash)

    # Demonstrate unique-slot truncation with k=3.
    truncated_ids = unique_slot_truncation(
        candidates, k=3, policy=RepresentativePolicy.DETERMINISTIC_LEXICOGRAPHIC
    )

    # Demonstrate representative policy preserves stronger contract evidence.
    groups = group_candidates_by_canonical_ast(
        candidates, policy=RepresentativePolicy.DETERMINISTIC_LEXICOGRAPHIC
    )

    return {
        "prompt_hash": prompt_hash,
        "pool_size": len(candidates),
        "arm_reports": {name: report.to_dict() for name, report in arms.items()},
        "truncated_ids_k3": list(truncated_ids),
        "group_count": len(groups),
        "representative_ids": [g.selected_representative_id for g in groups],
        "group_disagreements": {
            "hard": sum(1 for g in groups if g.has_hard_disagreement),
            "semantic": sum(1 for g in groups if g.has_semantic_disagreement),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SLM-130 EFS3-05 canonical AST dedup wiring/fixture harness",
        exit_on_error=False,
    )
    parser.add_argument(
        "--fixture",
        action="store_true",
        help="Run the built-in tiny synthetic fixture and compare all arms",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Path to write the JSON report",
    )
    try:
        args = parser.parse_args(argv)
    except (argparse.ArgumentError, SystemExit):
        return 2

    if not args.fixture:
        print("error: --fixture is currently the only supported mode", file=sys.stderr)
        return 2

    result = _run_fixture()
    result["version_stamp"] = build_version_stamp(
        "harness.experiments.canonical_ast_dedup",
    )
    result["run_date"] = _now()
    result["claim_class"] = "wiring / fixture only"

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True))
        print(f"wrote {args.out}")
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
