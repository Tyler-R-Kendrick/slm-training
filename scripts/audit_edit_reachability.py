#!/usr/bin/env python3
"""Audit edit-algebra reachability, canonical invariance, and transition certificates.

Examples:
  python -m scripts.audit_edit_reachability --describe
  python -m scripts.audit_edit_reachability --fixtures
  python -m scripts.audit_edit_reachability --target "root = Stack([n0], \"column\")\nn0 = TextContent(\":x\")" --slots :x
  python -m scripts.audit_edit_reachability --corpus src/slm_training/resources/test_seeds.jsonl --limit 5
  python -m scripts.audit_edit_reachability --emit-bridges --explain-unreachable
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.dsl.solver.topology_adapter import TopologyAdapterConfig
from slm_training.harnesses.experiments.slm188_edit_algebra import (
    EXPERIMENT_ID,
    MATRIX_SET,
    MATRIX_VERSION,
    EditReachabilityReport,
    build_fixture_codec,
    render_markdown,
    run_edit_reachability_fixture,
)
from slm_training.versioning import build_version_stamp

_DESIGN_JSON = "docs/design/iter-slm188-edit-algebra-20260721.json"
_DESIGN_MD = "docs/design/iter-slm188-edit-algebra-20260721.md"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _today_yyyymmdd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _describe_schema() -> str:
    return """\
SLM-188 edit-algebra reachability audit schema

CanonicalEdit fields:
  edit_id, action, node_id, production_id, arity, slot_id,
  preconditions, affected_node_ids, inverse_action, dependency_footprint,
  cost, coverage_tier.

TransitionCertificateV1 fields:
  schema, source_fingerprint, target_fingerprint, edit, source_program,
  target_program, verifier_profile, verifier_accepted, verifier_detail,
  version_pins, certificate_digest.

EditReachabilityCase fields:
  case_id, source_seed_id, target_id, target_program, target_fingerprint,
  result, path_length, edits, certificates, nodes_expanded, max_frontier,
  stop_reason, verifier_replay_ok, canonical_invariant_ok,
  alpha_invariant_ok, slot_invariant_ok.

EditReachabilityReport fields:
  schema, matrix_set, matrix_version, experiment_id, run_id, status,
  claim_class, hypothesis, falsifier, cases, invariance_results,
  n_cases, n_reachable, n_unreachable_complete, n_unknown_budget,
  n_unsupported, n_invariance_ok, disposition, disposition_rationale,
  honest_caveats, version_stamp, timestamp.

Audit scope:
  - Bounded BFS over topology EXPAND/KEEP/DELETE edits from a minimal seed.
  - Canonical target matching via D2 canonicalizer fingerprints.
  - TransitionCertificateV1 emitted and replay-verified for every reachable path.
  - Canonical idempotence, alpha-renaming, slot permutation, and commutativity checks.
  - v0.5/state/query/action targets are reported as unsupported_pack_feature.
  - No model, GPU, or checkpoint involvement.

Claim class: wiring / fixture only.
"""


def _load_corpus(path: Path, limit: int | None) -> list[tuple[str, str, list[str]]]:
    """Load (target_id, openui, placeholders) records from a JSONL corpus."""
    targets: list[tuple[str, str, list[str]]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            target_id = str(record.get("id", f"corpus_{len(targets)}"))
            source = str(record.get("openui", ""))
            placeholders = record.get("placeholders") or []
            targets.append((target_id, source, [str(p) for p in placeholders]))
            if limit is not None and len(targets) >= limit:
                break
    return targets


def _build_payload(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    if args.mode == "plan-only":
        payload: dict[str, Any] = {
            "schema": "EditReachabilityReportV1",
            "matrix_set": MATRIX_SET,
            "matrix_version": MATRIX_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "run_id": f"{EXPERIMENT_ID}-{_today_yyyymmdd()}",
            "status": "plan_only",
            "claim_class": "wiring",
            "hypothesis": run_edit_reachability_fixture.__doc__ or "",
            "falsifier": (
                "A bounded fixture search finds a supported canonical target that is "
                "unreachable from the minimal seed, or a transition certificate that "
                "does not replay, or a failed canonical invariance check."
            ),
            "cases": [],
            "invariance_results": [],
            "n_cases": 0,
            "n_reachable": 0,
            "n_unreachable_complete": 0,
            "n_unknown_budget": 0,
            "n_unsupported": 0,
            "n_invariance_ok": 0,
            "disposition": "inconclusive",
            "disposition_rationale": "Plan-only manifest; run --mode fixture to execute.",
            "honest_caveats": [
                "Plan-only: no reachability search was executed.",
                "Real bridge coverage needs the standard solver budget and corpus.",
            ],
            "version_stamp": build_version_stamp(
                "harness.experiments",
                "harness.experiments.slm188_edit_algebra",
                "dsl.solver.topology",
            ),
            "timestamp": _now(),
        }
        command = "python -m scripts.audit_edit_reachability --mode plan-only"
        return payload, command

    config = TopologyAdapterConfig(
        topology_max_nodes=args.max_nodes,
        topology_max_active=args.max_active,
        topology_max_arity=args.max_arity,
        topology_max_depth=args.max_depth,
    )
    codec = build_fixture_codec()

    targets: list[tuple[str, str, list[str]]] = []
    if args.target:
        slots = [s.strip() for s in args.slots.split(",")] if args.slots else []
        targets.append(("cli_target", args.target, slots))
    elif args.corpus:
        targets = _load_corpus(args.corpus, args.limit)
    else:
        # Default fixture targets from the harness.
        targets = [
            ("hero", _HERO, [":hero.title", ":hero.body"]),
            ("text_only", 'root = Stack([blurb], "column")\nblurb = TextContent(":page.blurb")', [":page.blurb"]),
        ]

    report = run_edit_reachability_fixture(
        codec=codec,
        config=config,
        targets=targets,
        seed_index=args.seed_index,
        max_edits=args.max_edits,
        run_id=f"{EXPERIMENT_ID}-{_today_yyyymmdd()}",
    )
    report.to_json(args.output_dir / "slm188_edit_algebra_report.json")
    payload = report.to_dict()
    command = "python -m scripts.audit_edit_reachability --fixtures"
    return payload, command


_HERO = 'root = Stack([hero], "column")\nhero_title = TextContent(":hero.title")\nhero_body = TextContent(":hero.body")\nhero = Card([hero_title, hero_body])'


def _build_markdown(payload: dict[str, Any], command: str) -> str:
    status = payload.get("status", "fixture")
    if status == "plan_only":
        lines = [
            "# SLM-188 (FFE1-02): edit-algebra reachability plan",
            "",
            "**Claim class:** wiring / fixture only",
            "",
            f"**Run date:** {_today_yyyymmdd()}",
            "",
            "**Machine-readable result:** [`iter-slm188-edit-algebra-20260721.json`](iter-slm188-edit-algebra-20260721.json)",
            "",
            "This is a plan-only manifest. The reachability engine, transition "
            "certificate schema, and invariance checks are wired; run `--mode fixture` "
            "to execute the CPU-only audit.",
            "",
            "## Hypothesis",
            "",
            "The canonical topology edit algebra can reach every supported fixture target "
            "from a minimal seed within declared bounds, and canonical invariance holds.",
            "",
            "## Falsifier",
            "",
            "A bounded fixture search finds a supported target that is unreachable, a "
            "certificate that does not replay, or a failed invariance check.",
            "",
            "## Exact command",
            "",
            f"```bash\n{command}\n```",
            "",
        ]
        return "\n".join(lines)

    report = EditReachabilityReport.from_dict(payload)
    return render_markdown(report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SLM-188 FFE1-02 edit-algebra reachability, invariance, and certificate audit",
        exit_on_error=False,
    )
    parser.add_argument(
        "--mode",
        choices={"plan-only", "fixture", "describe"},
        default="plan-only",
        help="Run mode: plan-only writes the manifest; fixture runs the CPU audit.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for run artifacts (default: outputs/runs/slm188-edit-algebra-<YYYYMMDD>)",
    )
    parser.add_argument(
        "--target",
        help="Single canonical OpenUI target to audit (overrides default fixtures).",
    )
    parser.add_argument(
        "--slots",
        help="Comma-separated slot contract for --target (e.g. ':a,:b').",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        help="JSONL corpus of records with 'openui' and 'placeholders' fields.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of corpus records to audit.",
    )
    parser.add_argument(
        "--seed-index",
        type=int,
        default=0,
        help="Index of the minimal seed to start from (default: 0).",
    )
    parser.add_argument(
        "--max-edits",
        type=int,
        default=12,
        help="Bounded BFS edit budget (default: 12).",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=64,
        help="Maximum topology nodes (default: 64).",
    )
    parser.add_argument(
        "--max-active",
        type=int,
        default=64,
        help="Maximum active topology nodes (default: 64).",
    )
    parser.add_argument(
        "--max-arity",
        type=int,
        default=4,
        help="Maximum component arity (default: 4).",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=4,
        help="Maximum topology depth (default: 4).",
    )
    parser.add_argument(
        "--emit-bridges",
        action="store_true",
        help="Write emitted transition certificates to bridges.jsonl.",
    )
    parser.add_argument(
        "--explain-unreachable",
        action="store_true",
        help="Print per-case diagnostics for unreachable/unknown targets.",
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
        help="Override path for the design JSON.",
    )
    parser.add_argument(
        "--design-md",
        type=Path,
        default=None,
        help="Override path for the design markdown.",
    )
    try:
        args = parser.parse_args(argv)
    except (argparse.ArgumentError, SystemExit):
        return 2

    if args.mode == "describe":
        print(_describe_schema())
        return 0

    args.output_dir = (
        args.output_dir
        or Path(f"outputs/runs/slm188-edit-algebra-{_today_yyyymmdd()}")
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    payload, command = _build_payload(args)
    payload["timestamp"] = _now()

    run_json = args.output_dir / "slm188_edit_algebra_report.json"
    run_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    if args.emit_bridges and args.mode == "fixture":
        bridges_path = args.output_dir / "bridges.jsonl"
        with bridges_path.open("w", encoding="utf-8") as fh:
            for case in payload.get("cases", ()):
                for cert in case.get("certificates", ()):
                    fh.write(json.dumps(cert, sort_keys=True, default=str) + "\n")

    if args.explain_unreachable and args.mode == "fixture":
        for case in payload.get("cases", ()):
            if case.get("result") in {"unreachable_complete", "unknown_budget"}:
                print(f"{case['case_id']}: {case['result']} — {case.get('stop_reason', '')}")

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
            command_line += f" --output-dir {args.output_dir}"
        md_path.write_text(_build_markdown(payload, command_line), encoding="utf-8")

    print(str(run_json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
