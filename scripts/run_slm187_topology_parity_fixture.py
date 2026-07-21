#!/usr/bin/env python3
"""Run the SLM-187 (FFE1-01) topology solver/runtime transition parity fixture.

Example:
  python -m scripts.run_slm187_topology_parity_fixture --mode plan-only
  python -m scripts.run_slm187_topology_parity_fixture --mode fixture
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.dsl.solver.topology_adapter import TopologyAdapterConfig
from slm_training.harnesses.experiments.slm187_topology_parity import (
    EXPERIMENT_ID,
    MATRIX_SET,
    MATRIX_VERSION,
    TopologyParityReport,
    build_fixture_codec,
    render_markdown,
    run_topology_parity_fixture,
)
from slm_training.versioning import build_version_stamp

_DESIGN_JSON = "docs/design/iter-slm187-topology-parity-20260721.json"
_DESIGN_MD = "docs/design/iter-slm187-topology-parity-20260721.md"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _today_yyyymmdd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _describe_schema() -> str:
    return """\
SLM-187 topology solver/runtime parity fixture schema

TopologyStateV2 fields:
  tree, problem_id, pack_id, constraint_version, output_kind, slot_inventory,
  bounds, phase, tree_fingerprint, state_fingerprint.

TopologyTransitionTuple fields:
  node_id, action, production_id, arity, slot_id.

TopologyParityCase fields:
  case_id, description, state_v2, solver_domain, runtime_domain,
  runtime_filtered_domain, solver_only, runtime_only, shared, parity_ok,
  terminal_text, terminal_valid, terminal_error.

TopologyParityReport fields:
  schema, matrix_set, matrix_version, experiment_id, run_id, status,
  claim_class, hypothesis, falsifier, cases, n_cases, n_parity_ok,
  n_runtime_only, n_solver_only, disposition, disposition_rationale,
  honest_caveats, version_stamp, timestamp.

Fixture scope:
  - 8 deterministic fixture trees covering active root, statement,
    component, list, leaf slot binding, fragment marker, max-depth leaf,
    and sibling choices.
  - Torch-free solver domain from derive_topology_holes.
  - Torch-free runtime proposal mirror of grammar_diffusion._decode_one.
  - Parity check over EXPAND/KEEP edits that participate in solver filtering.
  - Terminal serialization + DSL parser validation for solved trees.
  - No live model, GPU, or checkpoint involvement.

Claim class: wiring / fixture only.
"""


def _build_payload(mode: str, output_dir: Path, seed: int) -> tuple[dict[str, Any], str]:
    if mode == "plan-only":
        payload: dict[str, Any] = {
            "schema": "TopologyParityReportV1",
            "matrix_set": MATRIX_SET,
            "matrix_version": MATRIX_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "run_id": f"{EXPERIMENT_ID}-{_today_yyyymmdd()}",
            "status": "plan_only",
            "claim_class": "wiring",
            "hypothesis": run_topology_parity_fixture.__doc__ or "",
            "falsifier": (
                "An exhaustive fixture over bounded topology states finds a "
                "runtime-committable EXPAND or KEEP edit missing from the solver domain."
            ),
            "cases": [],
            "n_cases": 0,
            "n_parity_ok": 0,
            "n_runtime_only": 0,
            "n_solver_only": 0,
            "disposition": "inconclusive",
            "disposition_rationale": "Plan-only manifest; run --mode fixture to execute.",
            "honest_caveats": [
                "Plan-only: no fixture trees were evaluated.",
                "Real runtime parity under topology_verified_solver=True requires a trained model.",
            ],
            "version_stamp": build_version_stamp(
                "harness.experiments",
                "harness.experiments.slm187_topology_parity",
                "dsl.solver.topology",
            ),
            "timestamp": _now(),
        }
        command = "python -m scripts.run_slm187_topology_parity_fixture --mode plan-only"
        return payload, command

    report = run_topology_parity_fixture(
        codec=build_fixture_codec(),
        config=TopologyAdapterConfig(
            topology_max_nodes=8,
            topology_max_active=8,
            topology_max_arity=3,
            topology_max_depth=4,
        ),
        slot_inventory=[":hero.title", ":hero.body"],
        output_kind="document",
        seed=seed,
        run_id=f"{EXPERIMENT_ID}-{_today_yyyymmdd()}",
    )
    report.to_json(output_dir / "slm187_topology_parity_report.json")
    payload = report.to_dict()
    command = "python -m scripts.run_slm187_topology_parity_fixture --mode fixture"
    return payload, command


def _build_markdown(payload: dict[str, Any], command: str) -> str:
    status = payload.get("status", "fixture")
    if status == "plan_only":
        lines = [
            "# SLM-187 (FFE1-01): topology solver/runtime parity plan",
            "",
            "**Claim class:** wiring / fixture only",
            "",
            f"**Run date:** {_today_yyyymmdd()}",
            "",
            "**Machine-readable result:** [`iter-slm187-topology-parity-20260721.json`](iter-slm187-topology-parity-20260721.json)",
            "",
            "This is a plan-only manifest. The parity oracle, V2 state carrier, and "
            "fixture trees are wired; run `--mode fixture` to execute the CPU-only comparison.",
            "",
            "## Hypothesis",
            "",
            "The topology finite-domain solver state carries enough resolved tree and "
            "context information to reconstruct every successor, and the solver's "
            "enumerated edit domain contains every runtime-committable EXPAND/KEEP edit.",
            "",
            "## Falsifier",
            "",
            "An exhaustive fixture finds a runtime-committable EXPAND or KEEP edit "
            "missing from the solver domain.",
            "",
            "## Exact command",
            "",
            f"```bash\n{command}\n```",
            "",
        ]
        return "\n".join(lines)

    report = TopologyParityReport.from_dict(payload)
    return render_markdown(report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SLM-187 FFE1-01 topology solver/runtime transition parity fixture",
        exit_on_error=False,
    )
    parser.add_argument(
        "--mode",
        choices={"plan-only", "fixture", "describe"},
        default="plan-only",
        help="Run mode: plan-only writes the manifest; fixture runs the CPU comparison.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for run artifacts (default: outputs/runs/slm187-topology-parity-<YYYYMMDD>)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Deterministic seed for fixture generation (default: 0).",
    )
    parser.add_argument(
        "--write-design-docs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write design docs in fixture mode (default: True).",
    )
    parser.add_argument(
        "--design-json",
        type=Path,
        default=None,
        help="Override path for the design JSON (default: docs/design/iter-slm187-topology-parity-<YYYYMMDD>.json).",
    )
    parser.add_argument(
        "--design-md",
        type=Path,
        default=None,
        help="Override path for the design markdown (default: docs/design/iter-slm187-topology-parity-<YYYYMMDD>.md).",
    )
    try:
        args = parser.parse_args(argv)
    except (argparse.ArgumentError, SystemExit):
        return 2

    if args.mode == "describe":
        print(_describe_schema())
        return 0

    output_dir = (
        args.output_dir
        or Path(f"outputs/runs/slm187-topology-parity-{_today_yyyymmdd()}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    payload, command = _build_payload(args.mode, output_dir, args.seed)
    payload["timestamp"] = _now()

    run_json = output_dir / "slm187_topology_parity_report.json"
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
        command_line = command
        if args.output_dir is not None:
            command_line += f" --output-dir {output_dir}"
        md_path.write_text(_build_markdown(payload, command_line), encoding="utf-8")

    print(str(run_json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
