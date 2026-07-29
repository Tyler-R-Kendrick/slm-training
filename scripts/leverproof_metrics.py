#!/usr/bin/env python3
"""Export evidence and run the Lean-backed LeverProof checker."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from slm_training.harnesses.experiments.verified_metrics import (
    VerifiedMetricError,
    verify_metric_certificate,
    write_metric_evidence,
)
from slm_training.levers import INTERRUPT_AFTER_SECONDS


def _candidate_rows(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("candidates"), list):
            values = payload["candidates"]
        elif isinstance(payload, list):
            values = payload
        elif isinstance(payload, dict):
            values = [payload]
        else:
            raise VerifiedMetricError(f"{path} must contain candidate objects")
        if any(not isinstance(value, dict) for value in values):
            raise VerifiedMetricError(f"{path} contains a non-object candidate")
        rows.extend(values)
    return rows


def _cmd_export(args: argparse.Namespace) -> int:
    path = write_metric_evidence(
        args.out,
        run_id=args.run_id,
        evidence_bundle_path=args.evidence_bundle,
        feature_flags_path=args.feature_flags,
        campaign_manifest_path=args.campaign_manifest,
        cold_requests=args.cold_requests,
        warm_requests=args.warm_requests,
        candidates=_candidate_rows(args.candidate_json),
    )
    print(path)
    return 0


def _cmd_certify(args: argparse.Namespace) -> int:
    destination = args.certificate
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=destination.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        with temporary.open("w", encoding="utf-8") as output:
            completed = subprocess.run(
                [str(args.leverproof_bin), "check", str(args.evidence)],
                stdout=output,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=INTERRUPT_AFTER_SECONDS,
            )
        if completed.returncode != 0:
            raise VerifiedMetricError(
                "LeverProof rejected evidence: "
                + (completed.stderr.strip() or "no diagnostic")
            )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    verify_metric_certificate(
        evidence_path=args.evidence,
        certificate_path=destination,
        checker=args.leverproof_bin,
    )
    print(destination)
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    certificate = verify_metric_certificate(
        evidence_path=args.evidence,
        certificate_path=args.certificate,
        expected_campaign_manifest_sha256=args.campaign_manifest_sha256,
        expected_selected_candidate=args.selected_candidate,
        checker=args.leverproof_bin,
    )
    print(
        json.dumps(
            {
                "verified": True,
                "selected_candidate": certificate["selected_candidate"],
            },
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    export = subparsers.add_parser("export")
    export.add_argument("--run-id", required=True)
    export.add_argument("--evidence-bundle", type=Path, required=True)
    export.add_argument("--feature-flags", type=Path, required=True)
    export.add_argument("--campaign-manifest", type=Path)
    export.add_argument("--cold-requests", type=int, required=True)
    export.add_argument("--warm-requests", type=int, required=True)
    export.add_argument("--candidate-json", type=Path, action="append", required=True)
    export.add_argument("--out", type=Path, required=True)
    export.set_defaults(func=_cmd_export)

    certify = subparsers.add_parser("certify")
    certify.add_argument("--evidence", type=Path, required=True)
    certify.add_argument("--certificate", type=Path, required=True)
    certify.add_argument("--leverproof-bin", type=Path, required=True)
    certify.set_defaults(func=_cmd_certify)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--evidence", type=Path, required=True)
    verify.add_argument("--certificate", type=Path, required=True)
    verify.add_argument("--leverproof-bin", type=Path, required=True)
    verify.add_argument("--campaign-manifest-sha256")
    verify.add_argument("--selected-candidate")
    verify.set_defaults(func=_cmd_verify)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except VerifiedMetricError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    raise SystemExit(main())
