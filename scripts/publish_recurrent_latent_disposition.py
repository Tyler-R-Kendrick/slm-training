"""Publish or verify the fail-closed SLM-236 RSC disposition."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from slm_training.harnesses.experiments.slm236_recurrent_latent_disposition import (
    DEFAULT_JSON,
    DEFAULT_MARKDOWN,
    RecurrentLatentDispositionV1,
    build_report,
    render_markdown,
    validate_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    report = build_report(root)
    json_path, markdown_path = root / DEFAULT_JSON, root / DEFAULT_MARKDOWN
    if args.check:
        committed = RecurrentLatentDispositionV1.from_dict(json.loads(json_path.read_text()))
        report = replace(report, evidence_cutoff_commit=committed.evidence_cutoff_commit, version_stamp=committed.version_stamp)
        if validate_report(committed, root) or committed.to_dict() != report.to_dict() or markdown_path.read_text() != render_markdown(committed):
            return 1
        return 0
    json_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(render_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
