#!/usr/bin/env python3
"""Run SLM-403's bounded local typed-policy control matrix."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import quote

import torch

from slm_training.data.flow.operator_policy_corpus import (
    OperatorPolicyRowV1,
    build_operator_policy_corpus,
    materialize_operator_policy_selection,
)
from slm_training.dsl.operators.contracts import _fingerprint
from slm_training.dsl.schema import load_jsonl
from slm_training.evals.agentv import publish_agentv_evaluation
from slm_training.evals.cap2_operator import (
    build_frozen_cap2_suite,
    oracle_prediction,
    score_cap2_predictions,
)
from slm_training.harnesses.experiments.typed_operator_policy import (
    TYPED_OPERATOR_POLICY_HEAD_FAMILIES,
    TypedOperatorPolicyHeadFamily,
    TypedOperatorPolicyExampleV1,
    TypedOperatorPolicyScorer,
    controlled_partial_coverage_report,
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
CAP2_MANIFEST = ROOT / "src/slm_training/resources/evals/cap2_operator_v2.json"


@dataclass(frozen=True)
class _RuntimePolicyRow:
    row: OperatorPolicyRowV1
    trace: Any
    authority_resolver: Any


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
) -> tuple[list[TypedOperatorPolicyExampleV1], dict[str, Any], dict[str, _RuntimePolicyRow]]:
    rows = []
    reports = []
    runtime_rows: dict[str, _RuntimePolicyRow] = {}

    def collect(trace, collapse, authority_resolver) -> None:
        built, report = build_operator_policy_corpus(
            trace=trace,
            collapse=collapse,
            authority_resolver=authority_resolver,
            split=split,
            max_combinations_per_operator=max_combinations_per_operator,
        )
        rows.extend(built)
        runtime_rows.update(
            {
                row.row_id: _RuntimePolicyRow(row, trace, authority_resolver)
                for row in built
            }
        )
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
    }, runtime_rows


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


def _cap2_transition_score(
    *,
    arm: dict[str, Any],
    suite: dict[str, Any],
    runtime_rows: dict[str, _RuntimePolicyRow],
    held_out_evidence: dict[str, Any],
    max_combinations_per_operator: int,
) -> dict[str, Any]:
    """Score selected policy rows on CAP2 v2, oracle-replaying other strata."""
    predictions = {case["case_id"]: oracle_prediction(case) for case in suite["cases"]}
    corpus_rows = [
        json.loads(line)
        for line in Path(held_out_evidence["corpus"]["records_path"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    case_by_application = {
        str(row["legal_action"]["application_id"]): f"transition:{row['example_id']}"
        for row in corpus_rows
        if row["target_view"] == "dual" and row["outcome"] == "success"
    }
    case_by_id = {case["case_id"]: case for case in suite["cases"]}
    materialized: list[dict[str, Any]] = []
    for prediction in arm["predictions"]:
        runtime = runtime_rows[prediction["row_id"]]
        case_id = case_by_application.get(runtime.row.accepted_application_id)
        if case_id is None:
            continue
        if prediction["selected_action_row"] is None:
            predictions[case_id] = {}
            materialized.append({"row_id": runtime.row.row_id, "case_id": case_id, "deferred": True})
            continue
        selected_arguments = tuple(
            (str(slot_id), int(reference_row))
            for slot_id, reference_row in prediction["selected_argument_rows"]
        )
        applied = materialize_operator_policy_selection(
            trace=runtime.trace,
            row=runtime.row,
            authority_resolver=runtime.authority_resolver,
            selected_action_row=int(prediction["selected_action_row"]),
            selected_argument_rows=selected_arguments,
            max_combinations_per_operator=max_combinations_per_operator,
        )
        case = case_by_id[case_id]
        exact = applied.application_id == case["evidence"]["application_id"]
        application = applied.result.application
        assert application is not None and applied.result.state is not None
        arguments = [argument.to_dict() for argument in application.arguments]
        predictions[case_id] = {
            "schema": "cap2_operator_prediction/v1",
            "case_id": case_id,
            "accepted_legal_action_mass": 1.0,
            "operator_id": runtime.row.policy_input["action_rows"][prediction["selected_action_row"]]["operator_id"],
            "argument_fingerprint": _fingerprint({"schema": "cap2_argument_target/v1", "arguments": arguments}),
            "result_ast": applied.result.state.source,
            "effect_fingerprint": application.proof.effect_fingerprint if application.proof is not None else "",
            "locality_violations": case["gold"]["locality_violations"] if exact else -1,
            "unintended_edits": 0 if exact else -1,
            "final_state_id": case["gold"]["final_state_id"] if exact else applied.result.state.state_digest,
            "operator_count": 1,
            "telemetry": {"active_nodes": 1, "node_passes": 1, "remask_phases": 0, "model_calls": prediction["model_forwards"], "compiler_calls": 1, "verifier_calls": 1, "latency_ms": 0.0, "peak_memory_bytes": 0},
        }
        materialized.append({**applied.to_dict(), "case_id": case_id, "exact_gold_application": exact})
    return {"score": score_cap2_predictions(suite, predictions), "materialized": materialized, "suite_identity": {"suite_version": suite["suite_version"], "suite_hash": suite["suite_hash"], "suite_n": len(suite["cases"])}}


def _run_arm(
    *,
    head_family: TypedOperatorPolicyHeadFamily,
    name: str,
    train: Sequence[TypedOperatorPolicyExampleV1],
    held_out: Sequence[TypedOperatorPolicyExampleV1],
    seed: int,
    steps: int,
    learning_rate: float,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    model = TypedOperatorPolicyScorer.from_examples(
        train, dim=16, head_family=head_family
    )
    history = []
    complete_train = [
        row for row in train if row.view.coverage.value == "complete"
    ]
    if name == "zero":
        _zero(model)
    elif name == "enabled":
        if not complete_train:
            return {
                "head_family": head_family,
                "arm": name,
                "seed": seed,
                "skipped": "no_complete_train_rows",
            }
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
            return {
                "head_family": head_family,
                "arm": name,
                "seed": seed,
                "skipped": "no_complete_train_rows",
            }
        history = train_typed_operator_policy(
            model, shuffled, steps=steps, learning_rate=learning_rate
        )
    elif name != "random":
        raise ValueError(f"unknown arm {name}")
    result = _evaluate(model, held_out)
    return {
        "head_family": head_family,
        "arm": name,
        "seed": seed,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "loss": history,
        **result,
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# {report['run']['label']} ({report['issue']})",
        "",
        "Status: bounded local measured result; not a ship claim",
        "",
        "## Matrix",
        "",
        "| Head | Arm | Seed | Action | Action + arguments | Singleton forwards | Partial forced |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in report["matrix"]:
        if arm.get("skipped"):
            lines.append(
                f"| `{arm['head_family']}` | `{arm['arm']}` | {arm['seed']} | - | - | - | - |"
            )
            continue
        lines.append(
            f"| `{arm['head_family']}` | `{arm['arm']}` | {arm['seed']} | {arm['action_accuracy']:.3f} | "
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
    parser.add_argument("--issue", default="SLM-403")
    parser.add_argument("--label", default="DSH3-28 typed dynamic operator policy")
    parser.add_argument(
        "--corpus-work-dir",
        type=Path,
        required=True,
        help="ignored local corpus artifacts; durable report and AgentV stay in --output-dir",
    )
    parser.add_argument(
        "--head-families",
        default=",".join(TYPED_OPERATOR_POLICY_HEAD_FAMILIES),
        help="comma-separated typed policy heads; default runs the required five-family matrix",
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
        "--cap2-v2",
        action="store_true",
        help="materialize held-out policy selections into CAP2 v2 transition predictions",
    )
    parser.add_argument(
        "--controlled-partial",
        action="store_true",
        help="record SLM-404 evaluator-only 100/75/50/25 controlled PARTIAL slices",
    )
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
    head_families = tuple(
        family.strip() for family in args.head_families.split(",") if family.strip()
    )
    if not head_families or any(
        family not in TYPED_OPERATOR_POLICY_HEAD_FAMILIES for family in head_families
    ):
        parser.error(
            "--head-families must name one or more of "
            + ", ".join(TYPED_OPERATOR_POLICY_HEAD_FAMILIES)
        )
    if len(set(head_families)) != len(head_families):
        parser.error("--head-families must not repeat a family")
    stamp = build_version_stamp(
        "harness.experiments.typed_operator_policy",
        "data.flow.operator_policy_corpus",
        "model.quantization",
        "model.operator_policy_view",
        "evals.cap2_operator",
    )
    train, train_evidence, _train_runtime = _collect_rows(
        source=TRAIN_SOURCE,
        split="train",
        output_dir=corpus_work_dir / "train",
        stamp=stamp,
        max_roots=2,
        max_combinations_per_operator=args.max_combinations,
        record_id=args.train_record_id,
    )
    held_out, held_out_evidence, held_out_runtime = _collect_rows(
        source=HELD_OUT_SOURCE,
        split="dev",
        output_dir=corpus_work_dir / "held_out",
        stamp=stamp,
        max_roots=2 if args.cap2_v2 else 1,
        max_combinations_per_operator=args.max_combinations,
        record_id=args.held_out_record_id,
    )
    matrix = [
        _run_arm(
            head_family=head_family,  # type: ignore[arg-type]
            name=arm,
            train=train,
            held_out=held_out,
            seed=11,
            steps=args.steps,
            learning_rate=args.learning_rate,
        )
        for head_family in head_families
        for arm in ("zero", "random", "shuffled_labels", "enabled")
    ]
    cap2_suite = (
        build_frozen_cap2_suite(
            manifest_path=CAP2_MANIFEST,
            source_records_path=HELD_OUT_SOURCE,
            work_dir=corpus_work_dir / "cap2-suite",
            version_stamp=stamp,
        )
        if args.cap2_v2
        else None
    )
    cap2 = (
        {
            f"{item['head_family']}/{item['arm']}": _cap2_transition_score(
                arm=item,
                suite=cap2_suite,
                runtime_rows=held_out_runtime,
                held_out_evidence=held_out_evidence,
                max_combinations_per_operator=args.max_combinations,
            )
            for item in matrix
            if not item.get("skipped")
        }
        if args.cap2_v2
        else {}
    )
    by_head_arm = {
        (item["head_family"], item["arm"]): item for item in matrix
    }
    complete_train_rows = sum(
        row.view.coverage.value == "complete" for row in train
    )
    controlled_partial = (
        {
            row.row_id: controlled_partial_coverage_report(row)
            for row in held_out
            if row.view.coverage.value == "complete"
        }
        if args.controlled_partial
        else {}
    )
    complete_held_out_rows = sum(
        row.view.coverage.value == "complete" for row in held_out
    )
    controls = {}
    for head_family in head_families:
        torch.manual_seed(97)
        replay_model = TypedOperatorPolicyScorer.from_examples(
            train, dim=16, head_family=head_family  # type: ignore[arg-type]
        )
        replay_first = _evaluate(replay_model, held_out)
        replay_second = _evaluate(replay_model, held_out)
        enabled = by_head_arm[(head_family, "enabled")]
        controls[head_family] = {
            "enabled_vs_zero": (
                None
                if enabled.get("skipped")
                else _changes(by_head_arm[(head_family, "zero")], enabled)
            ),
            "enabled_vs_random": (
                None
                if enabled.get("skipped")
                else _changes(by_head_arm[(head_family, "random")], enabled)
            ),
            "prediction_replay_matched": (
                replay_first["predictions"] == replay_second["predictions"]
            ),
        }
    beneficial_heads = [
        head_family
        for head_family, control in controls.items()
        if (change := control["enabled_vs_zero"]) is not None
        and change["changed"] > 0
        and change["correct"] > change["wrong"]
    ]
    if args.controlled_partial and complete_held_out_rows == 0:
        decision = {
            "verdict": "reject",
            "reason": "no COMPLETE held-out shadow rows for the controlled coverage matrix",
        }
    elif complete_train_rows == 0:
        decision = {
            "verdict": "reject",
            "reason": "no COMPLETE local training rows; enabled and shuffled-label arms were not run",
        }
    elif not beneficial_heads:
        decision = {
            "verdict": "reject",
            "reason": "no typed head caused a beneficial enabled-versus-zero held-out choice change",
        }
    else:
        decision = {
            "verdict": "measured",
            "reason": (
                f"bounded local {len(head_families)}-head control matrix completed; "
                "this is not a ship decision"
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
                    f"{item['head_family']}/{item['arm']}": item.get("partial_forced")
                    for item in matrix
                },
            },
            {
                "id": "singleton-bypass",
                "criteria": "COMPLETE singleton decisions use zero scorer forwards.",
                "pass": all(
                    item.get("skipped") or item["singleton_forwards"] == 0 for item in matrix
                ),
                "result": {
                    f"{item['head_family']}/{item['arm']}": item.get("singleton_forwards")
                    for item in matrix
                },
            },
            {
                "id": "causal-controls-or-stop-rule",
                "criteria": "Causal controls have denominators when COMPLETE supervision exists; otherwise the no-COMPLETE stop rule is recorded.",
                "pass": (
                    (
                        complete_train_rows == 0
                        or all(
                        control["enabled_vs_zero"] is None
                        or control["enabled_vs_zero"]["eligible"] == len(held_out)
                        for control in controls.values()
                        )
                    )
                    and (bool(beneficial_heads) == (decision["verdict"] == "measured"))
                ),
                "result": {
                    "complete_train_rows": complete_train_rows,
                    "beneficial_heads": beneficial_heads,
                    "decision": decision,
                    **controls,
                },
            },
            {
                "id": "controlled-partial-defer",
                "criteria": "Controlled PARTIAL projections hide the complete shadow and never run a learned forced path.",
                "pass": (
                    (not args.controlled_partial or bool(controlled_partial))
                    and all(
                        item["shadow_hidden"]
                        and item["model_forwards"] == 0
                        and item["false_hard_eliminations"] == 0
                        for slices in controlled_partial.values()
                        for item in slices
                    )
                ),
                "result": controlled_partial,
            },
        ],
    )
    _rewrite_agentv_paths(output_dir)
    report = {
        "schema": "dsh3_28_typed_operator_policy_report/v1",
        "issue": args.issue,
        "run": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "device": "cpu",
            "backend": "typed_operator_feature_encoder",
            "steps": args.steps,
            "learning_rate": args.learning_rate,
            "max_combinations_per_operator": args.max_combinations,
            "head_families": list(head_families),
            "train_record_id": args.train_record_id,
            "held_out_record_id": args.held_out_record_id,
            "checkpoint": None,
            "ship_claim": False,
            "label": args.label,
            "controlled_partial": args.controlled_partial,
        },
        "counts": {
            "train_rows": len(train),
            "complete_train_rows": complete_train_rows,
            "held_out_rows": len(held_out),
            "complete_held_out_rows": complete_held_out_rows,
        },
        "matrix": matrix,
        "cap2": cap2,
        "controlled_partial": controlled_partial,
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
