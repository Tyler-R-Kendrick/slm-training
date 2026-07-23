#!/usr/bin/env python3
"""Publish or verify the authoritative SLM-213 SemanticFloorGateV1 closeout."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from slm_training.harnesses.experiments.semantic_floor_gate import (
    DEFAULT_GATE_PATH,
    SemanticFloorGateV1,
    build_semantic_floor_gate,
    canonical_gate_json,
    render_markdown,
    validate_gate_references,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=Path(DEFAULT_GATE_PATH))
    parser.add_argument("--markdown", type=Path, default=Path(DEFAULT_GATE_PATH).with_suffix(".md"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    json_path = args.json if args.json.is_absolute() else root / args.json
    markdown_path = args.markdown if args.markdown.is_absolute() else root / args.markdown

    existing: SemanticFloorGateV1 | None = None
    if json_path.is_file():
        existing = SemanticFloorGateV1.from_dict(
            json.loads(json_path.read_text(encoding="utf-8"))
        )
    gate = build_semantic_floor_gate(
        repo_root=root,
        source_commit=existing.source_commit if existing else None,
        generated_at=existing.generated_at if existing else None,
    )
    if existing is not None:
        gate = replace(gate, version_stamp=existing.version_stamp)
    failures = validate_gate_references(gate, repo_root=root)
    if failures:
        raise ValueError("; ".join(failures))

    expected_json = canonical_gate_json(gate)
    expected_markdown = render_markdown(gate)
    if args.check:
        if not json_path.is_file() or json_path.read_text(encoding="utf-8") != expected_json:
            print(f"stale semantic-floor JSON: {json_path}")
            return 1
        if not markdown_path.is_file() or markdown_path.read_text(encoding="utf-8") != expected_markdown:
            print(f"stale semantic-floor Markdown: {markdown_path}")
            return 1
        print(f"{gate.verdict} {gate.gate_hash}")
        return 0

    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(expected_json, encoding="utf-8")
    markdown_path.write_text(expected_markdown, encoding="utf-8")
    print(f"{gate.verdict} {gate.gate_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
