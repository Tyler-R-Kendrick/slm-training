#!/usr/bin/env python3
"""Export playground annotations into SFT seeds and preference pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from slm_training.harnesses.annotations import (
    DEFAULT_FEEDBACK_PATH,
    DEFAULT_HUMAN_PAIRS_PATH,
    DEFAULT_HUMAN_TRAIN_PATH,
    export_all,
    load_annotations,
)
from slm_training.harnesses.annotations.judge_audit import (
    freeze_blinded_pairs,
    import_blinded_labels,
)


def _jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: {exc}") from exc
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    status = sub.add_parser("status", help="Show annotation counts")
    status.add_argument("--feedback", type=Path, default=DEFAULT_FEEDBACK_PATH)

    export = sub.add_parser("export", help="Rebuild human train seeds + preference pairs")
    export.add_argument("--feedback", type=Path, default=DEFAULT_FEEDBACK_PATH)
    export.add_argument("--human-train", type=Path, default=DEFAULT_HUMAN_TRAIN_PATH)
    export.add_argument("--pairs", type=Path, default=DEFAULT_HUMAN_PAIRS_PATH)

    freeze = sub.add_parser(
        "freeze-audit", help="Freeze a blind pair-study package and private key"
    )
    freeze.add_argument("--source", type=Path, required=True)
    freeze.add_argument("--output-dir", type=Path, required=True)
    freeze.add_argument(
        "--private-key",
        type=Path,
        required=True,
        help="Unblinding key path outside the redacted package directory",
    )
    freeze.add_argument(
        "--training-records",
        type=Path,
        help="Pinned training JSONL used for the audit/train leakage check",
    )
    freeze.add_argument("--audit-id", required=True)
    freeze.add_argument("--seed", type=int, required=True)
    freeze.add_argument(
        "--allow-incomplete-sample",
        action="store_true",
        help="Wiring-only package; bypass the 90-110 pair campaign coverage gate",
    )

    import_audit = sub.add_parser(
        "import-audit", help="Validate and aggregate blinded audit labels"
    )
    import_audit.add_argument("--manifest", type=Path, required=True)
    import_audit.add_argument("--labels", type=Path, action="append", required=True)
    import_audit.add_argument("--output", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.cmd == "status":
        rows = load_annotations(args.feedback)
        ups = sum(1 for r in rows if r.rating == "up")
        downs = sum(1 for r in rows if r.rating == "down")
        payload = {
            "feedback": str(args.feedback),
            "total": len(rows),
            "up": ups,
            "down": downs,
        }
        print(json.dumps(payload, indent=2))
        return 0

    if args.cmd == "freeze-audit":
        if not args.allow_incomplete_sample and args.training_records is None:
            parser.error("freeze-audit requires --training-records for a campaign package")
        training_rows = _jsonl(args.training_records) if args.training_records else []
        training_ids = {
            str(row.get("record_id") or row.get("id") or "") for row in training_rows
        }
        if "" in training_ids:
            parser.error("every --training-records row needs record_id or id")
        result = freeze_blinded_pairs(
            _jsonl(args.source),
            audit_id=args.audit_id,
            seed=args.seed,
            output_dir=args.output_dir,
            private_key_path=args.private_key,
            enforce_campaign_coverage=not args.allow_incomplete_sample,
            training_record_ids=training_ids,
            training_records_sha256=(
                hashlib.sha256(args.training_records.read_bytes()).hexdigest()
                if args.training_records
                else None
            ),
        )
        print(json.dumps(result, indent=2))
        return 0

    if args.cmd == "import-audit":
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        public_path = args.manifest.parent / "blind_pairs.jsonl"
        pair_ids = {str(row["pair_id"]) for row in _jsonl(public_path)}
        result = import_blinded_labels(
            args.labels,
            audit_id=str(manifest["audit_id"]),
            pair_ids=pair_ids,
            output_path=args.output,
        )
        print(json.dumps(result, indent=2))
        return 0

    result = export_all(
        feedback_path=args.feedback,
        human_train_path=args.human_train,
        pairs_path=args.pairs,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
