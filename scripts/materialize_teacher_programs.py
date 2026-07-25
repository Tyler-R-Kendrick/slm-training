#!/usr/bin/env python3
"""Materialize deep-admitted SLM-266 rows as a canonical local train snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.harnesses.experiments.teacher_programs import (
    AdmissionMode,
    AdmissionResultV1,
    TeacherRawArchiveRefV1,
    materialize_teacher_admission,
)
from slm_training.versioning import build_version_stamp


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--raw-archive-uri", required=True)
    parser.add_argument("--raw-archive-manifest-sha256", required=True)
    args = parser.parse_args(argv)
    payload = json.loads(args.admission.read_text(encoding="utf-8"))
    result = AdmissionResultV1(
        mode=AdmissionMode(payload["mode"]),
        accepted=tuple(payload.get("accepted") or ()),
        rejected=tuple(payload.get("rejected") or ()),
        manifest_hash=str(payload["manifest_hash"]),
    )
    manifest = materialize_teacher_admission(
        result,
        output_dir=args.out,
        dataset_id=args.dataset_id,
        raw_archive=TeacherRawArchiveRefV1(
            uri=args.raw_archive_uri,
            manifest_sha256=args.raw_archive_manifest_sha256,
        ),
    )
    manifest["version_stamp"] = build_version_stamp(
        "harness.experiments.slm266_teacher_programs"
    )
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"dataset_id": args.dataset_id, "records": manifest["records"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
