#!/usr/bin/env python3
"""Run one durable GPU train->checkpoint->eval reference path from a manifest.

This CLI composes the existing canonical entry points
(``scripts.hf_jobs_train``, ``scripts.remote_train``, ``scripts.train_model``,
``scripts.evaluate_model``, ``scripts.sync_checkpoints``) and persists the same
immutable :class:`AcceleratorRunManifestV1` through describe, dry-run, submit,
reconcile, evaluate, and local-smoke phases.

Examples
--------
    python -m scripts.run_gpu_reference validate --manifest slm262.json
    python -m scripts.run_gpu_reference describe --manifest slm262.json
    python -m scripts.run_gpu_reference submit --manifest slm262.json --provider hf_jobs --dry-run
    python -m scripts.run_gpu_reference local-smoke --manifest slm262.json --steps 5
    python -m scripts.run_gpu_reference evaluate --manifest slm262.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from slm_training.harnesses.experiments.slm262_gpu_reference import (
    AcceleratorRunManifestV1,
    adapter_for,
    build_default_manifest,
    run_local_smoke,
)


def _load_manifest(path: Path) -> AcceleratorRunManifestV1:
    if not path.is_file():
        raise FileNotFoundError(path)
    return AcceleratorRunManifestV1.load_json(path)


def _save_manifest(manifest: AcceleratorRunManifestV1, path: Path) -> None:
    path.write_text(manifest.to_json() + "\n", encoding="utf-8")


def cmd_validate(args: argparse.Namespace) -> int:
    manifest = _load_manifest(Path(args.manifest))
    errors = manifest.check_ready(require_gpu=args.require_gpu)
    print(json.dumps({"ready": not errors, "errors": errors}, indent=2))
    return 0 if not errors else 2


def cmd_describe(args: argparse.Namespace) -> int:
    manifest = _load_manifest(Path(args.manifest))
    print(json.dumps(manifest.describe(), indent=2, default=str))
    return 0


def cmd_submit(args: argparse.Namespace) -> int:
    manifest = _load_manifest(Path(args.manifest))
    provider = args.provider or manifest.provider
    adapter = adapter_for(provider)
    result = adapter.submit(manifest, dry_run=bool(args.dry_run))
    print(json.dumps(result, indent=2, default=str))
    if not result.get("ok"):
        return 3

    updates: dict[str, Any] = {
        "provider": provider,
        "provider_request_id": result.get("provider_request_id"),
        "provider_job_id": result.get("provider_job_id"),
        "timestamps": {
            **dict(manifest.timestamps),
            "submitted_at": result.get("timestamp_submitted"),
        },
        "notes": tuple(list(manifest.notes) + [f"submitted via {provider} (dry_run={result.get('dry_run')})"]),
    }
    if args.dry_run:
        updates["disposition"] = manifest.disposition or "inconclusive"
    updated = manifest.__class__.from_dict({**manifest.to_dict(), **updates})
    out = Path(args.output or args.manifest)
    _save_manifest(updated, out)
    print(json.dumps({"manifest_written": str(out)}, indent=2))
    return 0


def cmd_reconcile(args: argparse.Namespace) -> int:
    manifest = _load_manifest(Path(args.manifest))
    adapter = adapter_for(manifest.provider)
    status = adapter.status(manifest)
    print(json.dumps(status, indent=2, default=str))
    if not status.get("ok"):
        return 4
    updated = adapter.reconcile(manifest, status.get("payload", {}))
    if args.job_id:
        updated = updated.__class__.from_dict(
            {**updated.to_dict(), "provider_job_id": args.job_id}
        )
    out = Path(args.output or args.manifest)
    _save_manifest(updated, out)
    print(json.dumps({"manifest_written": str(out)}, indent=2))
    return 0


def cmd_logs(args: argparse.Namespace) -> int:
    manifest = _load_manifest(Path(args.manifest))
    adapter = adapter_for(manifest.provider)
    print(adapter.logs(manifest))
    return 0


def cmd_cancel(args: argparse.Namespace) -> int:
    manifest = _load_manifest(Path(args.manifest))
    adapter = adapter_for(manifest.provider)
    result = adapter.cancel(manifest)
    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("ok") else 5


def cmd_local_smoke(args: argparse.Namespace) -> int:
    manifest = _load_manifest(Path(args.manifest))
    report = run_local_smoke(
        manifest,
        steps=args.steps,
        resume_steps=args.resume_steps,
        train_version=args.train_version,
        device=args.device,
        context_backend=args.context_backend,
        run_root=args.run_root,
    )
    print(json.dumps(report, indent=2, default=str))
    if not report.get("ok"):
        return 6

    # Merge artifact hashes back into the manifest as local compatibility evidence.
    inventory = report.get("inventory", {})
    full_state = {
        name: meta
        for name, meta in inventory.items()
        if "full_state" in name.lower() or name == "last_full_state.pt"
    }
    checkpoint = {
        name: meta
        for name, meta in inventory.items()
        if name not in full_state
    }
    updated = manifest.__class__.from_dict(
        {
            **manifest.to_dict(),
            "checkpoint_inventory": checkpoint,
            "full_state_inventory": full_state,
            "timestamps": {
                **dict(manifest.timestamps),
                "local_smoke_at": report.get("run_id"),
            },
            "notes": tuple(
                list(manifest.notes)
                + [f"local CPU smoke completed: {report.get('run_id')}"]
            ),
        }
    )
    out = Path(args.output or args.manifest)
    _save_manifest(updated, out)
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    manifest = _load_manifest(Path(args.manifest))
    errors = manifest.check_ready()
    if errors:
        print(json.dumps({"ready": False, "errors": errors}, indent=2))
        return 2

    # Prefer the locally materialized serving checkpoint; fall back to the bucket prefix.
    checkpoint = None
    for name in ("last.pt", "best_ship_score.pt"):
        if name in manifest.checkpoint_inventory:
            candidates = list(Path("outputs/runs").glob(f"*/checkpoints/{name}"))
            if candidates:
                checkpoint = candidates[0]
                break
    if checkpoint is None:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "no local checkpoint inventory; run local-smoke or reconcile a remote run first",
                },
                indent=2,
            )
        )
        return 7

    cmd = [
        sys.executable,
        "-m",
        "scripts.evaluate_model",
        "--run-id",
        manifest.run_id,
        "--checkpoint",
        str(checkpoint),
        "--device",
        args.device,
        "--ship-gates",
    ]
    if args.suites:
        cmd.extend(["--suites", args.suites])
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    print(proc.stdout[-4000:])
    if proc.stderr:
        print(proc.stderr[-2000:], file=sys.stderr)

    run_dir = Path("outputs/runs") / manifest.run_id
    gates_path = run_dir / "gates.json"
    eval_refs = list(manifest.evaluation_report_refs)
    if gates_path.is_file():
        eval_refs.append(str(gates_path))
    updated = manifest.__class__.from_dict(
        {
            **manifest.to_dict(),
            "evaluation_report_refs": tuple(eval_refs),
            "notes": tuple(
                list(manifest.notes)
                + [f"evaluate_model exit_code={proc.returncode} on {checkpoint.name}"]
            ),
        }
    )
    out = Path(args.output or args.manifest)
    _save_manifest(updated, out)
    return proc.returncode


def cmd_init(args: argparse.Namespace) -> int:
    """Create a default SLM-262 manifest from the current repo state."""
    manifest = build_default_manifest(
        args.run_id,
        provider=args.provider,
        data_snapshot_id=args.data_snapshot_id,
        data_snapshot_sha=args.data_snapshot_sha,
        eval_snapshot_id=args.eval_snapshot_id,
        eval_snapshot_sha=args.eval_snapshot_sha,
        target_decisions=args.target_decisions,
        train_version=args.train_version,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    _save_manifest(manifest, out)
    print(json.dumps({"manifest_written": str(out)}, indent=2))
    return 0


def _add_manifest_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", required=True, help="Path to AcceleratorRunManifestV1 JSON")
    parser.add_argument("--output", default=None, help="Where to write the updated manifest")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="Create a default manifest")
    init.add_argument("--run-id", required=True)
    init.add_argument("--provider", default="dry_run", choices=("hf_jobs", "remote_pod", "dry_run"))
    init.add_argument("--data-snapshot-id", default="UNKNOWN")
    init.add_argument("--data-snapshot-sha", default="UNKNOWN")
    init.add_argument("--eval-snapshot-id", default="UNKNOWN")
    init.add_argument("--eval-snapshot-sha", default="UNKNOWN")
    init.add_argument("--target-decisions", type=int, default=50_000)
    init.add_argument("--train-version", default="e530_visible_semantic_roles_r1_20260719")
    init.add_argument("--output", required=True)
    init.set_defaults(func=cmd_init)

    val = sub.add_parser("validate", help="Check manifest readiness")
    _add_manifest_arg(val)
    val.add_argument("--require-gpu", action="store_true")
    val.set_defaults(func=cmd_validate)

    desc = sub.add_parser("describe", help="Show the execution plan")
    _add_manifest_arg(desc)
    desc.set_defaults(func=cmd_describe)

    submit = sub.add_parser("submit", help="Submit or dry-run a provider job")
    _add_manifest_arg(submit)
    submit.add_argument("--provider", default=None, choices=("hf_jobs", "remote_pod", "dry_run"))
    submit.add_argument("--dry-run", action="store_true")
    submit.set_defaults(func=cmd_submit)

    reconcile = sub.add_parser("reconcile", help="Reconcile provider status into the manifest")
    _add_manifest_arg(reconcile)
    reconcile.add_argument("--job-id", default=None)
    reconcile.set_defaults(func=cmd_reconcile)

    logs = sub.add_parser("logs", help="Fetch provider logs")
    _add_manifest_arg(logs)
    logs.set_defaults(func=cmd_logs)

    cancel = sub.add_parser("cancel", help="Cancel the provider job")
    _add_manifest_arg(cancel)
    cancel.set_defaults(func=cmd_cancel)

    smoke = sub.add_parser("local-smoke", help="CPU-only compatibility smoke + resume check")
    _add_manifest_arg(smoke)
    smoke.add_argument("--steps", type=int, default=5)
    smoke.add_argument("--resume-steps", type=int, default=2)
    smoke.add_argument("--train-version", default="e530_visible_semantic_roles_r1_20260719")
    smoke.add_argument("--device", default="cpu")
    smoke.add_argument("--context-backend", default="scratch")
    smoke.add_argument(
        "--run-root",
        default="outputs/runs",
        help="Directory under which the smoke run directory is created",
    )
    smoke.set_defaults(func=cmd_local_smoke)

    evaluate = sub.add_parser("evaluate", help="Run full eval on the persisted checkpoint")
    _add_manifest_arg(evaluate)
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--suites", default=None, help="Comma-separated suite names")
    evaluate.set_defaults(func=cmd_evaluate)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
