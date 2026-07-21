#!/usr/bin/env python3
"""CLI for SLM-195 (FFE3-04): solver-only semantic ceiling harness.

Examples
--------
    python -m scripts.run_slm195_solver_semantic_ceiling init --run-id slm195_demo --output slm195.json
    python -m scripts.run_slm195_solver_semantic_ceiling describe --manifest slm195.json
    python -m scripts.run_slm195_solver_semantic_ceiling exact --manifest slm195.json --output report.json
    python -m scripts.run_slm195_solver_semantic_ceiling budget-grid --manifest slm195.json --output report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.harnesses.experiments.slm195_solver_semantic_ceiling import (
    ARM_NAMES,
    SolverCeilingManifestV1,
    SolverCeilingReport,
    build_default_manifest,
    run_ceiling,
)


def _load_manifest(path: Path) -> SolverCeilingManifestV1:
    if not path.is_file():
        raise FileNotFoundError(path)
    return SolverCeilingManifestV1.load_json(path)


def _save_report(report: SolverCeilingReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    report.write_json(path)


def cmd_init(args: argparse.Namespace) -> int:
    manifest = build_default_manifest(
        args.run_id,
        random_seed=args.random_seed,
        max_wall_seconds=args.max_wall_seconds,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_json(out)
    print(json.dumps({"manifest_written": str(out)}, indent=2))
    return 0


def cmd_describe(args: argparse.Namespace) -> int:
    manifest = _load_manifest(Path(args.manifest))
    print(json.dumps(manifest.describe(), indent=2, default=str))
    return 0


def _run_and_emit(manifest: SolverCeilingManifestV1, output_path: Path | None) -> int:
    errors = manifest.check_ready()
    if errors:
        print(json.dumps({"ready": False, "errors": errors}, indent=2))
        return 2
    try:
        report = run_ceiling(manifest)
    except Exception as exc:  # pragma: no cover - defensive
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        return 3
    print(json.dumps(report.to_dict(), ensure_ascii=False, default=str))
    if output_path is not None:
        _save_report(report, output_path)
        print(json.dumps({"report_written": str(output_path)}, ensure_ascii=False))
    return 0


def cmd_exact(args: argparse.Namespace) -> int:
    manifest = _load_manifest(Path(args.manifest))
    # ``exact`` always runs the full set of arms against the default budgets.
    manifest = manifest.__class__.from_dict(
        {
            **manifest.to_dict(),
            "arms": ARM_NAMES,
            "budgets": (10, 100, 1000),
        }
    )
    return _run_and_emit(manifest, Path(args.output) if args.output else None)


def cmd_dev(args: argparse.Namespace) -> int:
    # Dev is currently a placeholder alias for exact; it includes the oracle arm.
    return cmd_exact(args)


def cmd_budget_grid(args: argparse.Namespace) -> int:
    manifest = _load_manifest(Path(args.manifest))
    return _run_and_emit(manifest, Path(args.output) if args.output else None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="Create a default manifest")
    init.add_argument("--run-id", required=True)
    init.add_argument("--random-seed", type=int, default=0)
    init.add_argument("--max-wall-seconds", type=int, default=180)
    init.add_argument("--output", required=True)
    init.set_defaults(func=cmd_init)

    describe = sub.add_parser("describe", help="Show the execution plan")
    describe.add_argument("--manifest", required=True)
    describe.set_defaults(func=cmd_describe)

    exact = sub.add_parser("exact", help="Run all arms against default budgets")
    exact.add_argument("--manifest", required=True)
    exact.add_argument("--output", required=True)
    exact.set_defaults(func=cmd_exact)

    dev = sub.add_parser("dev", help="Dev-mode alias for exact")
    dev.add_argument("--manifest", required=True)
    dev.add_argument("--output", required=True)
    dev.set_defaults(func=cmd_dev)

    grid = sub.add_parser("budget-grid", help="Run manifest arms × budgets")
    grid.add_argument("--manifest", required=True)
    grid.add_argument("--output", required=True)
    grid.set_defaults(func=cmd_budget_grid)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
