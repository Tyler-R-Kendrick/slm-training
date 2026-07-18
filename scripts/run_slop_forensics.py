"""CLI + evidence report for LDI3-02 structural-slop forensics (SLM-129).

Profiles a corpus of OpenUI programs (JSONL of ``{program_id, corpus,
prompt_group, source}``) or a built-in deterministic fixture, ranks
parent-over-baseline motifs with group-bootstrap statistics, and emits a JSON
report + a compact Markdown summary. **No model is updated, no ban list is
applied, and over-representation is diagnostic only** — not causal preference
evidence.

    python scripts/run_slop_forensics.py --fixture \\
        --out docs/design/ldi3-02-slop-forensics-report-20260718.json
    python scripts/run_slop_forensics.py --corpus programs.jsonl --out report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from slm_training.harnesses.quality.slop_forensics import (
    ProgramFeatures,
    extract_features,
    forensics_report,
    profile_corpora,
    rank_motifs,
)
from slm_training.versioning import build_version_stamp


def _fixture_features() -> list[ProgramFeatures]:
    """A frozen fixture where the 'parent' corpus over-uses a default skeleton and
    a filler n-gram vs the Gold/Silver baseline — purely illustrative, no model."""
    feats: list[ProgramFeatures] = []
    for i in range(6):
        feats.append(
            ProgramFeatures(f"parent{i}", "parent", f"grp{i % 3}",
                            surface_ngrams=("Stack▁spacing", "spacing▁md"),
                            skeleton_hash="default-skeleton", placeholders=("title",),
                            grammar_motifs=("open_Stack→close_early",))
        )
    for i in range(6):
        feats.append(
            ProgramFeatures(f"gold{i}", "gold_silver", f"grp{i % 3}",
                            surface_ngrams=("Text▁value",) if i else ("Stack▁spacing",),
                            skeleton_hash=f"varied-{i}", placeholders=("title", "body"))
        )
    for i in range(6):
        feats.append(
            ProgramFeatures(f"held{i}", "held_out", f"grp{i % 3}",
                            surface_ngrams=("Stack▁spacing",), skeleton_hash="default-skeleton")
        )
    return feats


def _load_corpus(path: Path) -> list[ProgramFeatures]:
    feats: list[ProgramFeatures] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        feats.append(
            extract_features(
                row["program_id"], row["corpus"], row["source"],
                dsl=row.get("dsl"), prompt_group=row.get("prompt_group", ""),
            )
        )
    return feats


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Structural-slop forensics report (SLM-129)",
        "",
        f"_{report['note']}_",
        "",
        f"Findings: **{report['finding_count']}** — by class: `{report['by_detector_class']}`",
        "",
        "| family | motif | log-odds | CI | support | class |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for f in report["top_findings"][:20]:
        lines.append(
            f"| {f['family']} | `{f['motif']}` | {f['log_odds']} | "
            f"[{f['ci_low']}, {f['ci_high']}] | {f['support']} | {f['detector_class']} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", action="store_true", help="use the built-in fixture corpus")
    parser.add_argument("--corpus", type=Path, default=None, help="JSONL of program sources")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap-iters", type=int, default=200)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--markdown-out", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.corpus is not None:
        feats = _load_corpus(args.corpus)
    elif args.fixture:
        feats = _fixture_features()
    else:
        parser.error("pass --fixture or --corpus")

    findings = rank_motifs(
        profile_corpora(feats),
        seed=args.seed,
        bootstrap_iters=args.bootstrap_iters,
        verifier_associated=None,
    )
    report = forensics_report(findings)
    report["version_stamp"] = build_version_stamp("harness.quality.slop_forensics")
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    if args.markdown_out is not None:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(_markdown(report), encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
