#!/usr/bin/env python3
"""Export evidence and run the Lean-backed LeverProof checker."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from slm_training.harnesses.experiments.verified_metrics import (
    IN_REPO_CHECKER,
    VerifiedMetricError,
    _checker_project_root,
    optimum_feedback,
    formal_process_budget,
    FormalProjectLock,
    ProcessOutcome,
    run_formal_process,
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
        expectation_manifest_path=args.expectations,
        observations_path=args.observations,
    )
    print(path)
    return 0


def _cmd_certify(args: argparse.Namespace) -> int:
    checker = Path(args.leverproof_bin or IN_REPO_CHECKER)
    if not checker.is_file():
        raise VerifiedMetricError(
            "in-repo LeverProof checker is unavailable; run "
            "`make -C src/leverproof_lean build`"
        )
    destination = args.certificate
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=destination.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        project_root = _checker_project_root(checker)
        total, _interrupt_after, _grace = formal_process_budget(
            INTERRUPT_AFTER_SECONDS
        )
        try:
            with FormalProjectLock(project_root, timeout_seconds=total) as lock:
                completed = run_formal_process(
                    [str(checker), "check", str(args.evidence)],
                    cwd=project_root,
                    timeout_seconds=max(0.001, total - lock.wait_seconds),
                )
        except TimeoutError as exc:
            raise VerifiedMetricError(
                f"LeverProof certification lock timed out: {exc}"
            ) from exc
        if completed.timed_out:
            raise VerifiedMetricError("LeverProof certification exceeded the run cap")
        if completed.outcome is ProcessOutcome.LAUNCH_FAILED:
            raise VerifiedMetricError(
                f"LeverProof certification could not start: {completed.launch_error}"
            )
        if completed.stdout_truncated:
            raise VerifiedMetricError(
                "LeverProof certification output exceeded the bounded capture"
            )
        temporary.write_text(completed.stdout, encoding="utf-8")
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
        checker=checker,
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
        require_band_certificate=args.require_band_certificate,
    )
    print(
        json.dumps(
            {
                "verified": True,
                "selected_candidate": certificate["selected_candidate"],
                **optimum_feedback(certificate),
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
    export.add_argument("--expectations", type=Path)
    export.add_argument("--observations", type=Path)
    export.add_argument("--out", type=Path, required=True)
    export.set_defaults(func=_cmd_export)

    certify = subparsers.add_parser("certify")
    certify.add_argument("--evidence", type=Path, required=True)
    certify.add_argument("--certificate", type=Path, required=True)
    certify.add_argument("--leverproof-bin", type=Path)
    certify.set_defaults(func=_cmd_certify)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--evidence", type=Path, required=True)
    verify.add_argument("--certificate", type=Path, required=True)
    verify.add_argument("--leverproof-bin", type=Path)
    verify.add_argument("--require-band-certificate", action="store_true")
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
