#!/usr/bin/env python3
"""Build/describe the checkpoint × decode-path compatibility manifest (EFS0-02).

``--list`` describes the three decode paths and required audit structure without
needing any checkpoint (the EFS "list/describe before run" requirement). Without
``--list`` it builds a versioned compatibility manifest from durable checkpoint
references (``<name>.ref.json`` + ``<name>.meta.json`` pairs under
``--reference-dir``) and validates it. With no references it emits an honest
*deferred* manifest rather than inventing cells — the ten-checkpoint audit is
blocked until frontier/diagnostic checkpoints are synced (SLM-103).

Exit 0 when the manifest validates (even if deferred); non-zero on validation
errors. See ``docs/design/checkpoint-provenance.md`` and the decode-invariance
iteration doc.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.harnesses.model_build.checkpoint_path_manifest import (
    build_compatibility_manifest,
    identity_from_reference_and_meta,
    validate_compatibility_manifest,
)
from slm_training.harnesses.model_build.checkpoint_reference import CheckpointReferenceV1
from slm_training.harnesses.model_build.decode_path import all_decode_paths


def describe_paths() -> dict:
    return {
        "decode_paths": [
            {
                "path_id": spec.path_id,
                "description": spec.description,
                "generation_entry": spec.generation_entry,
                "completion_kind": spec.completion_kind,
                "supported_model_families": list(spec.supported_model_families),
                "supported_output_codecs": list(spec.supported_output_codecs),
                "runtime_override_fields": list(spec.runtime_override_fields()),
                "fingerprint": spec.fingerprint,
            }
            for spec in all_decode_paths()
        ]
    }


def load_identities(reference_dir: Path) -> list:
    identities = []
    for ref_path in sorted(reference_dir.glob("*.ref.json")):
        meta_path = ref_path.with_name(ref_path.name[: -len(".ref.json")] + ".meta.json")
        if not meta_path.is_file():
            continue
        ref = CheckpointReferenceV1.load_json(ref_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        identities.append(identity_from_reference_and_meta(ref, meta))
    return identities


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list",
        action="store_true",
        help="Describe the decode paths and exit (no checkpoint required).",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=None,
        help="Directory of <name>.ref.json + <name>.meta.json checkpoint pairs.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Write manifest JSON here.")
    args = parser.parse_args(argv)

    if args.list:
        print(json.dumps(describe_paths(), indent=2))
        return 0

    identities = load_identities(args.reference_dir) if args.reference_dir else []
    manifest = build_compatibility_manifest(identities)
    errors = validate_compatibility_manifest(manifest)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "checkpoint_count": manifest["checkpoint_count"],
                "complete_blocks": manifest["complete_blocks"],
                "usable_for_audit": manifest["usable_for_audit"],
                "note": manifest["note"],
                "validation_errors": errors,
            },
            indent=2,
        )
    )
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
