#!/usr/bin/env python3
"""SLM-214 (NCS0-01): SpectralSnapshotV1 inspection CLI.

Examples:
  slm inspect spectral --describe
  slm inspect spectral --output-dir outputs/runs/slm214-spectral-snapshot-test
  slm inspect spectral-null --shape 128x128 --initializer gaussian --draws 50
  slm inspect spectral-compare --left a.json --right b.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from slm_training.harnesses.experiments.slm214_spectral_snapshot import (
    EXPERIMENT_ID,
    MATRIX_SET,
    MATRIX_VERSION,
    SpectralSnapshotReport,
    build_toy_model,
    render_markdown,
    run_spectral_snapshot_fixture,
    sample_null_summary,
)
from slm_training.versioning import build_version_stamp

_DESIGN_JSON = "docs/design/iter-slm214-spectral-snapshot-20260721.json"
_DESIGN_MD = "docs/design/iter-slm214-spectral-snapshot-20260721.md"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _today_yyyymmdd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _parse_shape(value: str) -> tuple[int, int]:
    try:
        rows, cols = value.lower().split("x")
        return int(rows), int(cols)
    except Exception as exc:  # noqa: BLE001
        raise argparse.ArgumentTypeError(f"shape must be ROWSxCOLS, got {value!r}") from exc


def _describe_schema() -> str:
    return """\
SLM-214 SpectralSnapshotV1 schema

SpectralSnapshotV1 fields (per storage identity):
  snapshot_version, estimator_version, backend_version,
  matrix_id, canonical_path, semantic_role,
  storage_identity, tied_aliases, shape, aspect_ratio, dtype, device,
  trainable, eligibility, ineligibility_reason,
  singular_values, lambda_max, frobenius_norm, spectral_norm,
  stable_rank, effective_rank, spectral_entropy,
  hill_alpha, hill_xmin, hill_tail_count,
  null_key, null_draws, null_mean_alpha, null_sd_alpha,
  alpha_z, randomized_esd_distance, warnings, elapsed_ms.

SpectralSnapshotReport fields:
  schema, matrix_set, matrix_version, experiment_id, run_id, status,
  claim_class, hypothesis, falsifier, snapshots,
  n_matrices, n_eligible, n_ineligible, n_randomized_null,
  total_elapsed_ms, disposition, disposition_rationale,
  honest_caveats, version_stamp, timestamp.

Claim class: wiring / fixture only. No model-quality or promotion claim.
"""


def _build_plan_only_payload(command: str) -> dict[str, Any]:
    return {
        "schema": "SpectralSnapshotReportV1",
        "matrix_set": MATRIX_SET,
        "matrix_version": MATRIX_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "run_id": f"{EXPERIMENT_ID}-{_today_yyyymmdd()}",
        "status": "plan_only",
        "claim_class": "wiring",
        "hypothesis": run_spectral_snapshot_fixture.__doc__ or "",
        "falsifier": _FALSIFIER,
        "snapshots": [],
        "n_matrices": 0,
        "n_eligible": 0,
        "n_ineligible": 0,
        "n_randomized_null": 0,
        "total_elapsed_ms": 0.0,
        "disposition": "inconclusive",
        "disposition_rationale": "Plan-only manifest; run `slm inspect spectral` to execute.",
        "honest_caveats": [
            "Plan-only: no spectral inspection was executed.",
            "Native PyTorch SVD backend; optional WeightWatcher not exercised.",
        ],
        "version_stamp": build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm214_spectral_snapshot",
        ),
        "timestamp": _now(),
    }


_FALSIFIER = (
    "Native Hill/MLE alpha estimates are unstable across estimator seeds, or "
    "calibrated null scores cannot distinguish synthetic controls, or tied "
    "storage aliases produce duplicate snapshots."
)


def _load_checkpoint(checkpoint: str | None) -> torch.nn.Module:
    if checkpoint is None:
        return build_toy_model()
    path = Path(checkpoint)
    if path.is_file():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(payload, torch.nn.Module):
            return payload
        state = payload.get("state_dict") or payload.get("model") or payload
        model = build_toy_model()
        model.load_state_dict(state, strict=False)
        return model
    try:
        from slm_training.models.twotower import TwoTowerModel

        return TwoTowerModel.from_pretrained(checkpoint, device="cpu")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Could not load checkpoint {checkpoint!r}: {exc}") from exc


def _cmd_spectral(args: argparse.Namespace) -> int:
    if args.mode == "plan-only":
        payload = _build_plan_only_payload("slm inspect spectral --mode plan-only")
    else:
        model = _load_checkpoint(args.checkpoint)
        report = run_spectral_snapshot_fixture(
            model,
            null_draws=args.null_draws,
            roles=args.roles.split(",") if args.roles else None,
            max_matrices=args.max_matrices,
            device=args.device,
            initializer_guess=args.initializer_guess,
            run_id=f"{EXPERIMENT_ID}-{_today_yyyymmdd()}",
        )
        report.to_json(args.output_dir / "slm214_spectral_report.json")
        payload = report.to_dict()

    run_json = args.output_dir / "slm214_spectral_report.json"
    run_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    if args.mode == "fixture" and args.write_design_docs:
        root = Path(__file__).resolve().parents[1]
        json_path = args.design_json or root / _DESIGN_JSON
        md_path = args.design_md or root / _DESIGN_MD
        json_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        md_path.write_text(
            render_markdown(SpectralSnapshotReport.from_dict(payload)),
            encoding="utf-8",
        )

    print(str(run_json))
    return 0


def _cmd_spectral_null(args: argparse.Namespace) -> int:
    rows, cols = args.shape
    summary = sample_null_summary(
        rows,
        cols,
        dtype=getattr(torch, args.dtype),
        initializer=args.initializer,
        draws=args.draws,
    )
    payload = {
        "schema": "SpectralNullSummaryV1",
        "matrix_set": MATRIX_SET,
        "matrix_version": MATRIX_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "run_id": f"{EXPERIMENT_ID}-null-{_today_yyyymmdd()}",
        "shape": [rows, cols],
        "initializer": args.initializer,
        "draws": args.draws,
        "null_key": summary["null_key"],
        "mean_alpha": summary["mean_alpha"],
        "sd_alpha": summary["sd_alpha"],
        "version_stamp": build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm214_spectral_snapshot",
        ),
        "timestamp": _now(),
    }
    out_dir = args.output_dir or Path(f"outputs/runs/slm214-spectral-null-{_today_yyyymmdd()}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "slm214_spectral_null_summary.json"
    out_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(str(out_path))
    return 0


def _cmd_spectral_compare(args: argparse.Namespace) -> int:
    left = json.loads(Path(args.left).read_text(encoding="utf-8"))
    right = json.loads(Path(args.right).read_text(encoding="utf-8"))
    left_snaps = {s["matrix_id"]: s for s in left.get("snapshots", [])}
    right_snaps = {s["matrix_id"]: s for s in right.get("snapshots", [])}
    common = sorted(set(left_snaps) & set(right_snaps))
    if not common:
        print("No common matrix IDs to compare.")
        return 1
    print("matrix_id | left hill α | right hill α | Δ rand-ESD")
    for mid in common:
        lha = left_snaps[mid].get("hill_alpha")
        rha = right_snaps[mid].get("hill_alpha")
        ldist = left_snaps[mid].get("randomized_esd_distance")
        rdist = right_snaps[mid].get("randomized_esd_distance")
        delta = (
            f"{abs((ldist or 0) - (rdist or 0)):.4f}"
            if ldist is not None and rdist is not None
            else "—"
        )
        print(
            f"{mid} | {lha if lha is not None else '—'} | "
            f"{rha if rha is not None else '—'} | {delta}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SLM-214 NCS0-01 SpectralSnapshotV1 weight-matrix inspection",
        exit_on_error=False,
    )
    subparsers = parser.add_subparsers(dest="command")

    # spectral
    p_spectral = subparsers.add_parser("spectral", help="Inspect a model or the toy fixture.")
    p_spectral.add_argument(
        "--mode",
        choices={"plan-only", "fixture"},
        default="fixture",
        help="Run mode (default: fixture).",
    )
    p_spectral.add_argument(
        "--checkpoint",
        help="Checkpoint path, HF id, or omit for the toy fixture.",
    )
    p_spectral.add_argument(
        "--null-draws",
        type=int,
        default=50,
        help="Number of null draws for calibration (default: 50).",
    )
    p_spectral.add_argument(
        "--roles",
        help="Comma-separated role filter (e.g. 'self_attn_q,mlp').",
    )
    p_spectral.add_argument(
        "--max-matrices",
        type=int,
        default=None,
        help="Cap the number of inspected matrices.",
    )
    p_spectral.add_argument(
        "--device",
        default="cpu",
        help="Torch device (default: cpu).",
    )
    p_spectral.add_argument(
        "--initializer-guess",
        default="gaussian",
        choices={"gaussian", "xavier_uniform", "kaiming_uniform"},
        help="Null initializer family to assume when exact metadata is unavailable.",
    )
    p_spectral.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for run artifacts.",
    )
    p_spectral.add_argument(
        "--write-design-docs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write design docs in fixture mode (default: True).",
    )
    p_spectral.add_argument("--design-json", type=Path, default=None)
    p_spectral.add_argument("--design-md", type=Path, default=None)

    # spectral-null
    p_null = subparsers.add_parser("spectral-null", help="Generate a null-cache summary for a shape.")
    p_null.add_argument("--shape", type=_parse_shape, required=True, help="Matrix shape as ROWSxCOLS.")
    p_null.add_argument(
        "--initializer",
        default="gaussian",
        choices={"gaussian", "xavier_uniform", "kaiming_uniform"},
    )
    p_null.add_argument("--draws", type=int, default=50)
    p_null.add_argument("--dtype", default="float32", choices={"float32", "float64"})
    p_null.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for the null summary.",
    )

    # spectral-compare
    p_compare = subparsers.add_parser("spectral-compare", help="Compare two spectral report JSONs.")
    p_compare.add_argument("--left", type=Path, required=True)
    p_compare.add_argument("--right", type=Path, required=True)

    # describe
    parser.add_argument(
        "--describe",
        action="store_true",
        help="Print the schema and exit.",
    )

    try:
        args = parser.parse_args(argv)
    except (argparse.ArgumentError, SystemExit):
        return 2

    if args.describe:
        print(_describe_schema())
        return 0

    if args.command in {"spectral", "spectral-null"} and getattr(args, "output_dir", None) is None:
        slug = "slm214-spectral-snapshot" if args.command == "spectral" else "slm214-spectral-null"
        args.output_dir = Path(f"outputs/runs/{slug}-{_today_yyyymmdd()}")
    if getattr(args, "output_dir", None) is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.command == "spectral":
        return _cmd_spectral(args)
    if args.command == "spectral-null":
        return _cmd_spectral_null(args)
    if args.command == "spectral-compare":
        return _cmd_spectral_compare(args)

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
