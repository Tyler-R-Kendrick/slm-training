#!/usr/bin/env python3
"""EFS4-04 campaign synthesizer.

Reads the preregistered Evidence-First Semantic SLM campaign manifest and every
committed ``docs/design/iter-*`` result JSON, validates the evidence contract,
builds the causal diagnosis / architecture dispositions / evidence graph, and
writes the machine-readable and human-readable synthesis reports.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from slm_training.harnesses.experiments.efs4_04_causal_synthesis import (
    EvidenceFirstSemanticSynthesisV1,
    build_default_campaign_manifest,
    load_manifest,
    load_result_manifests,
    render_dot,
    render_mermaid,
    save_manifest,
    save_synthesis,
    synthesize_campaign,
    validate_synthesis,
)


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("docs/design/evidence-first-semantic-slm-campaign-v1.json"),
        help="Path to the campaign manifest JSON (default: docs/design/evidence-first-semantic-slm-campaign-v1.json)",
    )
    parser.add_argument(
        "--docs-design",
        type=Path,
        default=Path("docs/design"),
        help="Directory containing committed result manifests (default: docs/design)",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Machine-readable synthesis output path (default: docs/design/iter-efs4-04-causal-synthesis-YYYYMMDD.json)",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=None,
        help="Human-readable Markdown output path (default: docs/design/iter-efs4-04-causal-synthesis-YYYYMMDD.md)",
    )
    parser.add_argument(
        "--graph-output",
        type=Path,
        default=None,
        help="Optional evidence-graph output stem (writes .mmd and .dot)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate and print errors without writing outputs",
    )
    parser.add_argument(
        "--describe",
        action="store_true",
        help="Print a summary of the loaded manifest and result matches",
    )
    parser.add_argument(
        "--write-default-manifest",
        action="store_true",
        help="Write the default campaign manifest if --manifest does not exist",
    )
    args = parser.parse_args(argv)

    # Default output paths.
    today = _today()
    out_json = args.out_json or Path(f"docs/design/iter-efs4-04-causal-synthesis-{today}.json")
    out_md = args.out_md or Path(f"docs/design/iter-efs4-04-causal-synthesis-{today}.md")

    # Load or create campaign manifest.
    if args.manifest.exists():
        manifest = load_manifest(args.manifest)
    elif args.write_default_manifest:
        manifest = build_default_campaign_manifest()
        save_manifest(manifest, args.manifest)
        print(f"Wrote default campaign manifest to {args.manifest}", file=sys.stderr)
        return 0
    else:
        print(
            f"Campaign manifest not found: {args.manifest} "
            f"(use --write-default-manifest to create it)",
            file=sys.stderr,
        )
        return 1

    # Load committed result manifests.
    try:
        result_manifests = load_result_manifests(args.docs_design)
    except FileNotFoundError as exc:
        print(f"Result directory not found: {exc}", file=sys.stderr)
        return 1

    if args.describe:
        from slm_training.harnesses.experiments.efs4_04_causal_synthesis import _matches

        print(f"Campaign: {manifest.campaign_id}")
        print(f"Hypotheses: {len(manifest.hypotheses)}")
        print(f"Result manifests loaded: {len(result_manifests)}")
        matched = 0
        for hyp in manifest.hypotheses:
            hits = [m.source_path for m in result_manifests if any(_matches(ref, m) for ref in hyp.expected_result_refs)]
            if hits:
                matched += 1
                print(f"  {hyp.hypothesis_id}: matched {hits}")
            else:
                print(f"  {hyp.hypothesis_id}: no match")
        print(f"Matched hypotheses: {matched}/{len(manifest.hypotheses)}")
        return 0

    # Synthesize.
    command = f"python -m scripts.synthesize_efs_campaign --manifest {args.manifest} --docs-design {args.docs_design}"
    synthesis = synthesize_campaign(manifest, result_manifests, generation_command=command)

    errors = validate_synthesis(synthesis)
    if errors:
        for err in errors:
            print(f"VALIDATION ERROR: {err}", file=sys.stderr)
        if args.validate_only:
            return 1
        # Non-fatal when writing: the report remains honest about unresolved items.
        print("Validation reported unresolved items; writing report anyway.", file=sys.stderr)

    if args.validate_only:
        print("Synthesis validates (no blocking errors).")
        return 0

    # Write outputs.
    save_synthesis(synthesis, out_json)
    out_md.write_text(render_markdown(synthesis), encoding="utf-8")
    print(f"Wrote synthesis JSON: {out_json}")
    print(f"Wrote synthesis Markdown: {out_md}")

    if args.graph_output:
        mmd_path = args.graph_output.with_suffix(".mmd")
        dot_path = args.graph_output.with_suffix(".dot")
        mmd_path.write_text(render_mermaid(synthesis.evidence_graph), encoding="utf-8")
        dot_path.write_text(render_dot(synthesis.evidence_graph), encoding="utf-8")
        print(f"Wrote evidence graph: {mmd_path}, {dot_path}")

    return 0


def render_markdown(synthesis: EvidenceFirstSemanticSynthesisV1) -> str:
    """Local re-export to avoid a second import in __main__."""
    from slm_training.harnesses.experiments.efs4_04_causal_synthesis import render_markdown as _render

    return _render(synthesis)


if __name__ == "__main__":
    raise SystemExit(main())
