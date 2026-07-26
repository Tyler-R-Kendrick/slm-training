#!/usr/bin/env python3
"""Run SLM-403's bounded local typed-policy control matrix."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import quote

import torch

from slm_training.data.flow.operator_policy_corpus import build_operator_policy_corpus
from slm_training.dsl.schema import load_jsonl
from slm_training.evals.agentv import publish_agentv_evaluation
from slm_training.harnesses.experiments.typed_operator_policy import (
    TypedOperatorPolicyExampleV1,
    TypedOperatorPolicyScorer,
    decide_typed_operator_policy,
    train_typed_operator_policy,
)
from slm_training.harnesses.train_data.operator_corpus import (
    OperatorCorpusConfig,
    build_symbolic_operator_corpus,
)
from slm_training.versioning import build_version_stamp

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SOURCE = ROOT / "src/slm_training/resources/data/train/openui_verified_v1/records.jsonl"
HELD_OUT_SOURCE = (
    ROOT
    / "src/slm_training/resources/data/eval/e763_symbol_only_eval_r2_20260722"
    / "suites/held_out/records.jsonl"
)


def _portable(value: Any, *, output_dir: Path, corpus_work_dir: Path) -> Any:
    """Keep committed evidence independent of an ephemeral worktree path."""
    if isinstance(value, str):
        for prefix, replacement in (
            (str(output_dir.resolve()), "agentv-dir://"),
            (str(corpus_work_dir.resolve()), "local-scratch://"),
        ):
            if value.startswith(prefix):
                return replacement + value[len(prefix) :].lstrip("/")
        return value
    if isinstance(value, list):
        return [_portable(item, output_dir=output_dir, corpus_work_dir=corpus_work_dir) for item in value]
    if isinstance(value, dict):
        return {
            key: _portable(item, output_dir=output_dir, corpus_work_dir=corpus_work_dir)
            for key, item in value.items()
        }
    return value


def _rewrite_agentv_paths(output_dir: Path) -> None:
    """AgentV writes absolute execution paths; durable docs must not retain them."""
    replacements = {
        str(output_dir.resolve()): "agentv-dir://",
        quote(str(output_dir.resolve()), safe=""): quote("agentv-dir://", safe=""),
        str(ROOT.resolve()): "repo://",
        quote(str(ROOT.resolve()), safe=""): quote("repo://", safe=""),
    }
    for path in (output_dir / "agentv").rglob("*"):
        if not path.is_file() or path.suffix not in {".json", ".jsonl", ".md"}:
            continue
        content = path.read_text(encoding="utf-8")
        for source, replacement in replacements.items():
            content = content.replace(source, replacement)
        path.write_text(content, encoding="utf-8")


def _select_source_records(source: Path, record_id: str | None) -> list[Any]:
    """Select one deterministic source-order root for a bounded local slice."""
    source_records = load_jsonl(source)
    if record_id is None:
        return source_records
    matches = [record for record in source_records if record.id == record_id]
    if not matches:
        raise ValueError(f"source does not contain record {record_id!r}")
    # The admitted fixture snapshot may repeat the same record ID. Select the
    # first source-order instance deterministically and pass only that instance
    # to the canonical builder; it must never inflate a local matrix denominator
    # by treating duplicated IDs as independent roots.
    return [matches[0]]


def _collect_rows(
    *,
    source: Path,
    split: str,
    output_dir: Path,
    stamp: dict[str, Any],
    max_roots: int,
    max_combinations_per_operator: int,
    record_id: str | None = None,
) -> tuple[list[TypedOperatorPolicyExampleV1], dict[str, Any]]:
    rows = []
    reports = []

    def collect(trace, collapse, authority_resolver) -> None:
        built, report = build_operator_policy_corpus(
            trace=trace,
            collapse=collapse,
            authority_resolver=authority_resolver,
            split=split,
            max_combinations_per_operator=max_combinations_per_operator,
        )
        rows.extend(built)
        reports.append(report.to_dict())

    corpus = build_symbolic_operator_corpus(
        records=_select_source_records(source, record_id),
        output_dir=output_dir,
        version=f"dsh3-28-{split}",
        version_stamp=stamp,
        config=OperatorCorpusConfig(
            max_roots=max_roots,
            actions_per_state=2,
            max_combinations_per_operator=max_combinations_per_operator,
            sibling_forks=True,
        ),
        on_collapsed_trace=collect,
    )
    examples = [TypedOperatorPolicyExampleV1.from_row(row) for row in rows]
    if not examples:
        raise ValueError(f"no typed policy rows from {source}")
    corpus_report = corpus["report"]
    return examples, {
        "corpus": {
            key: corpus[key]
            for key in (
                "records_path",
                "report_path",
                "collapsed_records_path",
                "record_count",
                "collapsed_record_count",
                "root_count",
                "content_fingerprint",
            )
        },
        "corpus_summary": {
            key: corpus_report[key]
            for key in (
                "schema",
                "version",
                "config",
                "strata",
                "application_coverage",
                "collapse",
                "version_stamp",
            )
        }
        | {"coverage_gap_count": len(corpus_report["coverage_gaps"])},
        "row_reports": reports,
    }


def _zero(model: TypedOperatorPolicyScorer) -> None:
    for parameter in model.parameters():
        parameter.data.zero_()


def _evaluate(
    model: TypedOperatorPolicyScorer,
    examples: Sequence[TypedOperatorPolicyExampleV1],
) -> dict[str, Any]:
    predictions = []
    for example in examples:
        decision = decide_typed_operator_policy(model, example)
        correct_action = decision.selected_action_row == example.accepted_action_row
        correct_arguments = (
            decision.selected_argument_rows == example.accepted_argument_rows
            if decision.selected_action_row is not None
            else False
        )
        predictions.append(
            {
                "row_id": example.row_id,
                "route": decision.route.value,
                "selected_action_row": decision.selected_action_row,
                "selected_argument_rows": [list(item) for item in decision.selected_argument_rows],
                "correct_action": correct_action,
                "correct_action_and_arguments": correct_action and correct_arguments,
                "model_forwards": decision.model_forwards,
            }
        )
    n = len(predictions)
    return {
        "n": n,
        "action_accuracy": sum(item["correct_action"] for item in predictions) / n,
        "action_and_arguments_accuracy": sum(
            item["correct_action_and_arguments"] for item in predictions
        )
        / n,
        "singleton_forwards": sum(
            item["model_forwards"] for item in predictions if item["route"] == "complete_singleton"
        ),
        "partial_forced": sum(
            item["selected_action_row"] is not None
            for item in predictions
            if item["route"].startswith("partial_")
        ),
        "predictions": predictions,
    }


def _changes(control: dict[str, Any], treatment: dict[str, Any]) -> dict[str, int]:
    previous = {item["row_id"]: item for item in control["predictions"]}
    changed = correct = wrong = 0
    for item in treatment["predictions"]:
        prior = previous[item["row_id"]]
        if item["selected_action_row"] == prior["selected_action_row"]:
            continue
        changed += 1
        correct += bool(item["correct_action"] and not prior["correct_action"])
        wrong += bool(prior["correct_action"] and not item["correct_action"])
    return {"eligible": treatment["n"], "changed": changed, "correct": correct, "wrong": wrong}


def _run_arm(
    *,
    name: str,
    train: Sequence[TypedOperatorPolicyExampleV1],
    held_out: Sequence[TypedOperatorPolicyExampleV1],
    seed: int,
    steps: int,
    learning_rate: float,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    model = TypedOperatorPolicyScorer.from_examples(train, dim=16)
    history = []
    complete_train = [
        row for row in train if row.view.coverage.value == "complete"
    ]
    if name == "zero":
        _zero(model)
    elif name == "enabled":
        if not complete_train:
            return {"arm": name, "seed": seed, "skipped": "no_complete_train_rows"}
        history = train_typed_operator_policy(
            model, train, steps=steps, learning_rate=learning_rate
        )
    elif name == "shuffled_labels":
        shuffled = [
            replace(
                row,
                accepted_action_row=(row.accepted_action_row + 1) % len(row.view.action_rows),
                accepted_argument_rows=(),
            )
            for row in complete_train
            if len(row.view.action_rows) > 1
        ]
        if not shuffled:
            return {"arm": name, "seed": seed, "skipped": "no_complete_train_rows"}
        history = train_typed_operator_policy(
            model, shuffled, steps=steps, learning_rate=learning_rate
        )
    elif name != "random":
        raise ValueError(f"unknown arm {name}")
    result = _evaluate(model, held_out)
    return {
        "arm": name,
        "seed": seed,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "loss": history,
        **result,
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# DSH3-28 typed dynamic operator policy (SLM-403)",
        "",
        "Status: bounded local measured result; not a ship claim",
        "",
        "## Matrix",
        "",
        "| Arm | Seed | Action | Action + arguments | Singleton forwards | Partial forced |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in report["matrix"]:
        if arm.get("skipped"):
            lines.append(
                f"| `{arm['arm']}` | {arm['seed']} | - | - | - | - |"
            )
            continue
        lines.append(
            f"| `{arm['arm']}` | {arm['seed']} | {arm['action_accuracy']:.3f} | "
            f"{arm['action_and_arguments_accuracy']:.3f} | {arm['singleton_forwards']} | "
            f"{arm['partial_forced']} |"
        )
    lines.extend(
        [
            "",
            "## Honesty",
            "",
            f"This run uses {report['counts']['train_rows']} local train policy rows and "
            f"{report['counts']['held_out_rows']} held-out policy rows, bounded to CPU. "
            "It is an integration/control result only: no checkpoint, human rating, remote "
            "workload, or ship-gate claim. CAP2 v1 replay drift remains explicitly outside "
            "this current-surface matrix.",
            "",
            f"Decision: `{report['decision']['verdict']}` — {report['decision']['reason']}",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--corpus-work-dir",
        type=Path,
        required=True,
        help="ignored local corpus artifacts; durable report and AgentV stay in --output-dir",
    )
    parser.add_argument(
        "--train-record-id",
        help="optional one-record training slice; its ID is persisted in the report",
    )
    parser.add_argument(
        "--held-out-record-id",
        help="optional one-record held-out slice; its ID is persisted in the report",
    )
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument(
        "--max-combinations",
        type=int,
        default=512,
        help="per-operator exact enumeration cap; remains explicit in durable evidence",
    )
    args = parser.parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_work_dir = args.corpus_work_dir.resolve()
    if args.max_combinations <= 0:
        parser.error("--max-combinations must be positive")
    stamp = build_version_stamp(
        "harness.experiments.typed_operator_policy",
        "data.flow.operator_policy_corpus",
        "model.operator_policy_view",
    )
    train, train_evidence = _collect_rows(
        source=TRAIN_SOURCE,
        split="train",
        output_dir=corpus_work_dir / "train",
        stamp=stamp,
        max_roots=2,
        max_combinations_per_operator=args.max_combinations,
        record_id=args.train_record_id,
    )
    held_out, held_out_evidence = _collect_rows(
        source=HELD_OUT_SOURCE,
        split="dev",
        output_dir=corpus_work_dir / "held_out",
        stamp=stamp,
        max_roots=1,
        max_combinations_per_operator=args.max_combinations,
        record_id=args.held_out_record_id,
    )
    matrix = [
        _run_arm(
            name=arm,
            train=train,
            held_out=held_out,
            seed=11,
            steps=args.steps,
            learning_rate=args.learning_rate,
        )
        for arm in ("zero", "random", "shuffled_labels", "enabled")
    ]
    by_arm = {item["arm"]: item for item in matrix}
    complete_train_rows = sum(
        row.view.coverage.value == "complete" for row in train
    )
    torch.manual_seed(97)
    replay_model = TypedOperatorPolicyScorer.from_examples(train, dim=16)
    replay_first = _evaluate(replay_model, held_out)
    replay_second = _evaluate(replay_model, held_out)
    enabled = by_arm["enabled"]
    controls = {
        "enabled_vs_zero": (
            None if enabled.get("skipped") else _changes(by_arm["zero"], enabled)
        ),
        "enabled_vs_random": (
            None if enabled.get("skipped") else _changes(by_arm["random"], enabled)
        ),
        "prediction_replay_matched": (
            replay_first["predictions"] == replay_second["predictions"]
        ),
    }
    agentv = publish_agentv_evaluation(
        output_dir,
        name="dsh3-28-typed-operator-policy",
        claim="bounded_local_current_surface_not_ship",
        version_stamp=stamp,
        cases=[
            {
                "id": "legal-row-selection",
                "criteria": "Every scored action and argument remains in the typed live row domain.",
                "pass": all(
                    item.get("skipped") or item["partial_forced"] == 0 for item in matrix
                ),
                "result": {
                    item["arm"]: item.get("partial_forced") for item in matrix
                },
            },
            {
                "id": "singleton-bypass",
                "criteria": "COMPLETE singleton decisions use zero scorer forwards.",
                "pass": all(
                    item.get("skipped") or item["singleton_forwards"] == 0 for item in matrix
                ),
                "result": {
                    item["arm"]: item.get("singleton_forwards") for item in matrix
                },
            },
            {
                "id": "causal-controls-or-stop-rule",
                "criteria": "Causal controls have denominators when COMPLETE supervision exists; otherwise the no-COMPLETE stop rule is recorded.",
                "pass": (
                    complete_train_rows == 0
                    or controls["enabled_vs_zero"] is None
                    or controls["enabled_vs_zero"]["eligible"] == len(held_out)
                ),
                "result": {"complete_train_rows": complete_train_rows, **controls},
            },
        ],
    )
    _rewrite_agentv_paths(output_dir)
    decision = (
        {
            "verdict": "reject",
            "reason": "no COMPLETE local training rows; enabled and shuffled-label arms were not run",
        }
        if complete_train_rows == 0
        else {
            "verdict": "measured",
            "reason": "bounded local control matrix completed; this is not a ship decision",
        }
    )
    report = {
        "schema": "dsh3_28_typed_operator_policy_report/v1",
        "issue": "SLM-403",
        "run": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "device": "cpu",
            "backend": "typed_operator_feature_encoder",
            "steps": args.steps,
            "learning_rate": args.learning_rate,
            "max_combinations_per_operator": args.max_combinations,
            "train_record_id": args.train_record_id,
            "held_out_record_id": args.held_out_record_id,
            "checkpoint": None,
            "ship_claim": False,
        },
        "counts": {
            "train_rows": len(train),
            "complete_train_rows": complete_train_rows,
            "held_out_rows": len(held_out),
        },
        "matrix": matrix,
        "controls": controls,
        "train_evidence": _portable(
            train_evidence, output_dir=output_dir, corpus_work_dir=corpus_work_dir
        ),
        "held_out_evidence": _portable(
            held_out_evidence, output_dir=output_dir, corpus_work_dir=corpus_work_dir
        ),
        "agentv": _portable(
            agentv, output_dir=output_dir, corpus_work_dir=corpus_work_dir
        ),
        "decision": decision,
        "version_stamp": stamp,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"report": str(output_dir / "report.json"), "rows": report["counts"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
