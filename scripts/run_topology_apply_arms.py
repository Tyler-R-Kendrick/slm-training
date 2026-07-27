#!/usr/bin/env python3
"""Run DSH3-16's topology-apply arm comparison (SLM-384)."""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

from slm_training.evals.agentv import publish_agentv_evaluation
from slm_training.harnesses.experiments.topology_apply_arms import (
    TopologyApplyArm,
    run_topology_apply_arms,
)
from slm_training.versioning import build_version_stamp


def _portable(value: Any, output_dir: Path) -> Any:
    prefix = str(output_dir.resolve())
    if isinstance(value, str) and value.startswith(prefix):
        return "output-dir://" + value[len(prefix) :].lstrip("/")
    if isinstance(value, list):
        return [_portable(item, output_dir) for item in value]
    if isinstance(value, dict):
        return {key: _portable(item, output_dir) for key, item in value.items()}
    return value


def _rewrite_agentv_paths(output_dir: Path) -> None:
    prefixes = {
        str(output_dir.resolve()): "output-dir://",
        quote(str(output_dir.resolve()), safe=""): quote("output-dir://", safe=""),
    }
    for path in (output_dir / "agentv").rglob("*"):
        if not path.is_file() or path.suffix not in {".json", ".jsonl", ".md"}:
            continue
        value = path.read_text(encoding="utf-8")
        for prefix, replacement in prefixes.items():
            value = value.replace(prefix, replacement)
        path.write_text(value, encoding="utf-8")


def _evidence_cases(result: dict[str, Any]) -> list[dict[str, Any]]:
    acceptance = result["acceptance"]
    totals = {arm: data["totals"] for arm, data in result["arms"].items()}
    topology = totals[TopologyApplyArm.TOPOLOGY_APPLY.value]
    recompute = totals[TopologyApplyArm.RECOMPUTE_ONLY.value]
    return [
        {
            "id": "matched-quality",
            "criteria": (
                "Legal terminal set, per-step typed trees, and action sequences "
                "are identical across recompute-only, hierarchical-head, and "
                "topology-apply arms."
            ),
            "pass": bool(
                acceptance["legal_terminal_set_unchanged"]
                and acceptance["matched_workload_identical_action_sequences"]
                and acceptance["matched_quality_every_step"]
            ),
            "result": {
                "terminal_digests": {
                    run["request_id"]: run["terminal_digest"]
                    for run in result["arms"][TopologyApplyArm.TOPOLOGY_APPLY.value][
                        "requests"
                    ]
                }
            },
        },
        {
            "id": "authority-preserved",
            "criteria": (
                "Exact singletons commit with zero model forwards in every arm, "
                "the default-off hierarchical head always DEFERs, and "
                "topology-apply never skips compiler expansions or verifier "
                "calls."
            ),
            "pass": bool(
                acceptance["exact_singleton_authority_preserved"]
                and acceptance["head_always_defers_off_mode"]
                and acceptance["zero_model_forwards_without_head"]
                and topology["compiler_expansions"] == recompute["compiler_expansions"]
                and topology["verifier_calls"] == recompute["verifier_calls"]
            ),
            "result": {
                "singleton_bypasses": {
                    arm: t["singleton_bypasses"] for arm, t in totals.items()
                },
                "compiler_expansions": {
                    arm: t["compiler_expansions"] for arm, t in totals.items()
                },
            },
        },
        {
            "id": "node-pass-work",
            "criteria": (
                "Topology-apply reduces typed-node re-materialization passes "
                "vs. recompute-only at matched quality (the preregistered "
                "execution-sparsity work metric)."
            ),
            "pass": bool(acceptance["node_pass_improvement_at_matched_quality"]),
            "result": {
                "node_passes": {arm: t["node_passes"] for arm, t in totals.items()},
                "direct_applies": topology["direct_applies"],
                "deferred_applies": topology["deferred_applies"],
            },
        },
        {
            "id": "honest-scope",
            "criteria": (
                "The disposition is fixture/wiring-labeled, makes no systems "
                "or wall-clock claim, and records rejection when the stop rule "
                "triggers."
            ),
            "pass": result["systems_claim"] is False
            and result["evidence_tier"] == "fixture_wiring"
            and result["disposition"]
            in ("execution_sparsity_supported", "execution_sparsity_rejected"),
            "result": {
                "disposition": result["disposition"],
                "elapsed_seconds": {
                    arm: data["elapsed_seconds"]
                    for arm, data in result["arms"].items()
                },
            },
        },
    ]


def _markdown(report: dict[str, Any]) -> str:
    result = report["result"]
    totals = {arm: data["totals"] for arm, data in result["arms"].items()}
    lines = [
        "# Iter: SLM-384 DSH3-16 verified operators as topology-diffusion edit actions",
        "",
        "Date: 2026-07-25",
        f"Status: measured; {result['disposition'].replace('_', '-')} (fixture/wiring)",
        "Scope: bounded CPU fixture arm comparison; no checkpoint, ship, or systems claim",
        "",
        "## Decision",
        "",
        "DSH3-16 tests whether applying compiler-verified operator edits directly to",
        "typed topology nodes reduces diffusion-native decode work at matched quality.",
        "Three matched arms ran over one deterministic request set",
        f"(max {result['max_steps']} steps/request, enumeration budget",
        f"{result['max_combinations_per_operator']} combinations/operator):",
        "",
        "- `RECOMPUTE_ONLY` -- ordinary lowering; full typed-tree re-materialization per edit.",
        "- `HIERARCHICAL_HEAD` -- the SLM-383 head in default `off` mode shadow-scores",
        "  every non-singleton decision (2 scorer forwards) and always DEFERs;",
        "  ordinary lowering applies. This measures head overhead at an unchanged workload.",
        "- `TOPOLOGY_APPLY` -- the identical action sequence, but exact legal",
        "  `ActionEffectV1` transitions map to topology expand/contract/move/rewrite",
        "  edits applied directly to typed nodes (`dsl/operators/topology_apply.py`);",
        "  non-exact or unmapped transitions defer to ordinary lowering.",
        "",
        "Forced singletons commit with zero model forwards in every arm; compiler",
        "expansions and pack-authority verifier calls are never skipped.",
        "",
        "## Arm totals (matched workload)",
        "",
        "| Arm | Steps | Node passes | Remask | Rewrite | Model forwards | Compiler expansions | Verifier calls | Direct | Deferred |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm, t in totals.items():
        lines.append(
            f"| `{arm}` | {t['steps']} | {t['node_passes']} | {t['remask_phases']} | "
            f"{t['rewrite_phases']} | {t['model_forwards']} | "
            f"{t['compiler_expansions']} | {t['verifier_calls']} | "
            f"{t['direct_applies']} | {t['deferred_applies']} |"
        )
    lines.extend(["", "## Wall-clock / memory (not a speed claim)", ""])
    lines.append("| Arm | Elapsed (s) | Peak memory (bytes) |")
    lines.append("| --- | ---: | ---: |")
    for arm, data in result["arms"].items():
        lines.append(
            f"| `{arm}` | {data['elapsed_seconds']:.2f} | {data['peak_memory_bytes']:,} |"
        )
    lines.extend(["", "## Acceptance", ""])
    for key, value in result["acceptance"].items():
        lines.append(f"- `{key}`: **{'pass' if value else 'fail'}**")
    lines.extend(
        [
            "",
            f"Disposition: **{result['disposition']}** (evidence tier "
            f"`{result['evidence_tier']}`, systems claim: {result['systems_claim']}).",
            "",
            result["scope_note"],
            "",
            result["enumeration_budget_note"],
            "",
            "The harness re-materializes authoritative trees every step to certify",
            "matched quality, so arm wall-clock is conservative *against*",
            "topology-apply; no wall-clock or systems claim is made, and serialized",
            "target compression is not reported as inference speed. A real systems",
            "claim would require preregistered node-pass/model-call or wall-clock",
            "improvement with exact hardware/workload identity on the neural decode",
            "path, which this fixture does not provide.",
            "",
            f"The run completed in {report['run']['elapsed_seconds']:.2f}s with peak process "
            f"memory {report['run']['peak_memory_bytes']:,} bytes. AgentV passed "
            f"{report['agentv']['summary']['passed']}/"
            f"{report['agentv']['summary']['total']} evidence cases with "
            "zero execution errors.",
            "",
            "No checkpoint was created, so the model card and README checkpoint summary",
            "do not change.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument(
        "--max-combinations-per-operator", type=int, default=128
    )
    args = parser.parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = build_version_stamp(
        "dsl.operators.topology_apply",
        "dsl.operators.legal_set",
        "dsl.operators.hierarchical_head",
        "harness.experiments.topology_apply_arms",
    )
    started = time.perf_counter()
    peak_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = run_topology_apply_arms(
        max_steps=args.max_steps,
        max_combinations_per_operator=args.max_combinations_per_operator,
    )
    peak_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_memory = max(peak_before, peak_after) * 1024
    elapsed = time.perf_counter() - started
    agentv = publish_agentv_evaluation(
        output_dir,
        name="dsh3-16-topology-apply-arms",
        claim="bounded_fixture_arm_comparison_wiring_not_ship",
        cases=_evidence_cases(result),
        version_stamp=stamp,
    )
    _rewrite_agentv_paths(output_dir)
    report = {
        "schema": "topology_apply_arms_report/v1",
        "run": {
            "issue": "SLM-384",
            "experiment_id": "DSH3-16",
            "device": "cpu",
            "backend": "operator_legal_set+topology_apply+hierarchical_head(off)",
            "max_wall_minutes": 15,
            "elapsed_seconds": elapsed,
            "peak_memory_bytes": peak_memory,
            "checkpoint": None,
            "ship_claim": False,
        },
        "result": result,
        "agentv": _portable(agentv, output_dir),
        "version_stamp": stamp,
    }
    args.report_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.summary_md.write_text(_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "issue": "SLM-384",
                "disposition": result["disposition"],
                "elapsed_seconds": elapsed,
                "report": str(args.report_json),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
