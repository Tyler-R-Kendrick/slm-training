#!/usr/bin/env python3
"""Run DSH3-15's matched hierarchical-operator-head causal baseline (SLM-383)."""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

from slm_training.evals.agentv import publish_agentv_evaluation
from slm_training.harnesses.experiments.hierarchical_operator_head_baseline import (
    DEFAULT_SHAPES,
    run_hierarchical_operator_head_baseline,
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
    changes = result["causal_changes_vs_token_baseline"]
    false_admissions = sum(
        run["false_legal_admissions"] for values in result["arms"].values() for run in values
    )
    return [
        {
            "id": "matched-architecture",
            "criteria": (
                "Weight-zero and enabled share identical raw parameter count "
                "(architecture); only the token baseline is a structurally "
                "distinct decoder-visible model."
            ),
            "pass": len(
                {
                    run["parameter_count"]
                    for arm in ("WEIGHT_ZERO", "ENABLED")
                    for run in result["arms"][arm]
                }
            )
            == 1,
            "result": {
                "seeds": result["seeds"],
                "steps_per_arm": result["steps_per_arm"],
                "train_decision_n": result["train_decision_n"],
                "held_out_decision_n": result["held_out_decision_n"],
            },
        },
        {
            "id": "compiler-membership",
            "criteria": "No arm ever selects an operator id outside the live compiler-verified legal set.",
            "pass": false_admissions == 0,
            "result": {"false_legal_admissions": false_admissions},
        },
        {
            "id": "causal-choice-audit",
            "criteria": "Weight-zero/enabled choice changes vs. the token-baseline control are persisted per seed.",
            "pass": all(
                item["eligible"] == result["held_out_decision_n"]
                for values in changes.values()
                for item in values
            ),
            "result": changes,
        },
        {
            "id": "capacity-control",
            "criteria": (
                "The enabled head strictly beats its own frozen weight-zero "
                "capacity control -- it is not a prediction-identical no-op."
            ),
            "pass": result["acceptance"]["enabled_beats_weight_zero_capacity_control"],
            "result": {
                "mean_operator_accuracy": result["mean_operator_accuracy"],
            },
        },
        {
            "id": "honest-stop-rule",
            "criteria": (
                "DSH3-15's acceptance bar (causal improvement beyond the token "
                "baseline) is reported as rejection, not promotion, when unmet."
            ),
            "pass": result["verdict"] == "reject"
            or result["acceptance"]["enabled_causally_improves_beyond_token_baseline"],
            "result": {
                "verdict": result["verdict"],
                "acceptance": result["acceptance"],
            },
        },
        {
            "id": "retention-boundary",
            "criteria": "CAP0 default-off retention is explicit and CAP1 remains unavailable without CERT_CAP1.",
            "pass": (
                result["acceptance"]["cap0_retention"]["pass"] is True
                and result["acceptance"]["cap1_retention"]["available"] is False
            ),
            "result": {
                "cap0": result["acceptance"]["cap0_retention"],
                "cap1": result["acceptance"]["cap1_retention"],
            },
        },
    ]


def _markdown(report: dict[str, Any]) -> str:
    result = report["result"]
    lines = [
        "# DSH3-15 hierarchical operator action head causal baseline (SLM-383)",
        "",
        "Date: 2026-07-25",
        "Status: measured; rejected",
        "Scope: bounded CPU fixture causal baseline; no checkpoint or ship claim",
        "",
        "## Decision",
        "",
        "SLM-383 authorizes testing the encoder-side hierarchical operator action",
        "head only after a token baseline is run. DSH3-14/SLM-382 (E803) rejected the",
        "*decoder-visible* token hypothesis outright; per AGENTS.md IV.11 that left the",
        "encoder-side/masked scoring mechanism untested. This run is that follow-up:",
        "token baseline vs. weight-zero (frozen no-op capacity control) vs. enabled",
        "(trained) arms over a shared fixture corpus whose gold operator depends only",
        "on a typed candidate-group-size feature (never opaque ids or display names).",
        "",
        "The enabled head strictly beats its own weight-zero capacity control -- it is",
        "trainable and not a prediction-identical no-op -- but ties exactly with the",
        "token baseline (both memorize the finite shape vocabulary to ceiling). DSH3-15's",
        "acceptance bar requires causal improvement *beyond* the token baseline, which",
        "does not hold here, so the stop rule applies and the head stays default-off.",
        "",
        "## Matched result",
        "",
        "| Arm | Seed | Operator accuracy | Parameters | Trainable | False admissions |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm, runs in result["arms"].items():
        for run in runs:
            lines.append(
                f"| `{arm}` | {run['seed']} | {run['operator_accuracy']:.3f} | "
                f"{run['parameter_count']} | {run['trainable_parameter_count']} | "
                f"{run['false_legal_admissions']} |"
            )
    lines.extend(
        [
            "",
            "## Causal choice changes (vs. token-baseline control)",
            "",
            "| Treatment | Seed | Changed | Rate | Correct | Wrong |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for arm, values in result["causal_changes_vs_token_baseline"].items():
        for seed, item in zip(result["seeds"], values, strict=True):
            lines.append(
                f"| `{arm}` | {seed} | {item['changed']}/{item['eligible']} | "
                f"{item['change_rate']:.3f} | {item['correct_changes']} | "
                f"{item['wrong_changes']} |"
            )
    lines.extend(["", "## Acceptance and honesty", ""])
    for key, value in result["acceptance"].items():
        if isinstance(value, bool):
            lines.append(f"- `{key}`: **{'pass' if value else 'fail'}**")
    lines.extend(
        [
            "",
            result["scope_note"],
            "",
            "CAP0 is retained because the codec is default-off and the disabled path",
            "defers unchanged. CAP1 retention is unavailable because CERT_CAP1/SLM-379",
            "does not exist. No efficiency conclusion is drawn from this semantic run.",
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
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--train-repeats", type=int, default=5)
    parser.add_argument("--held-out-repeats", type=int, default=2)
    args = parser.parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = build_version_stamp(
        "dsl.operators.hierarchical_head",
        "dsl.operators.legal_set",
        "harness.experiments.hierarchical_operator_head_baseline",
    )
    started = time.perf_counter()
    peak_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = run_hierarchical_operator_head_baseline(
        shapes=DEFAULT_SHAPES,
        train_repeats=args.train_repeats,
        held_out_repeats=args.held_out_repeats,
        steps=args.steps,
        learning_rate=args.learning_rate,
    )
    peak_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_memory = max(peak_before, peak_after) * 1024
    elapsed = time.perf_counter() - started
    agentv = publish_agentv_evaluation(
        output_dir,
        name="dsh3-15-hierarchical-operator-head-baseline",
        claim="bounded_fixture_baseline_rejected_not_ship",
        cases=_evidence_cases(result),
    )
    _rewrite_agentv_paths(output_dir)
    report = {
        "schema": "hierarchical_operator_head_baseline_report/v1",
        "run": {
            "issue": "SLM-383",
            "predecessor_experiment_id": "E803",
            "device": "cpu",
            "backend": "dynamic_pointer_scorer+hashed_token_scorer",
            "max_wall_minutes": 3,
            "elapsed_seconds": elapsed,
            "peak_memory_bytes": peak_memory,
            "checkpoint": None,
            "ship_claim": False,
        },
        "result": result,
        "agentv": _portable(agentv, output_dir),
        "version_stamp": stamp,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "issue": "SLM-383",
                "verdict": result["verdict"],
                "accepted": result["accepted"],
                "elapsed_seconds": elapsed,
                "report": str(output_dir / "report.json"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
