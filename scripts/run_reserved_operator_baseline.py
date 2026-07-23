#!/usr/bin/env python3
"""Run E803's matched reserved-operator token baseline."""

from __future__ import annotations

import argparse
import json
import resource
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

from slm_training.dsl.schema import load_jsonl
from slm_training.evals.agentv import publish_agentv_evaluation
from slm_training.harnesses.experiments.reserved_operator_baseline import (
    run_reserved_operator_baseline,
)
from slm_training.harnesses.train_data.operator_corpus import (
    OperatorCorpusConfig,
    build_symbolic_operator_corpus,
)
from slm_training.versioning import build_version_stamp

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SOURCE = ROOT / "src/slm_training/resources/train_seeds.jsonl"
HELD_OUT_SOURCE = (
    ROOT
    / "src/slm_training/resources/data/eval"
    / "e763_symbol_only_eval_r2_20260722/suites/held_out/records.jsonl"
)
CAP2_MANIFEST = ROOT / "src/slm_training/resources/evals/cap2_operator_v1.json"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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
    changes = result["causal_changes_vs_result_ast_only"]
    false_admissions = sum(
        run["false_legal_admissions"]
        for values in result["arms"].values()
        for run in values
    )
    return [
        {
            "id": "matched-arms",
            "criteria": "All three arms use identical seeds, steps, examples, and parameter capacity.",
            "pass": len(
                {
                    run["parameter_count"]
                    for values in result["arms"].values()
                    for run in values
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
            "criteria": "No model-visible target is admitted outside the compiler-generated candidate set.",
            "pass": false_admissions == 0,
            "result": {"false_legal_admissions": false_admissions},
        },
        {
            "id": "causal-choice-audit",
            "criteria": "Enabled/disabled choice changes and their correct/wrong direction are persisted per seed.",
            "pass": all(
                item["eligible"] == result["held_out_decision_n"]
                for values in changes.values()
                for item in values
            ),
            "result": changes,
        },
        {
            "id": "honest-stop-rule",
            "criteria": "A failed semantic acceptance contract is reported as rejection, not promotion.",
            "pass": result["verdict"] == "reject" and not result["accepted"],
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
        "# E803 reserved discrete-operator token baseline (SLM-382)",
        "",
        "Date: 2026-07-23",
        "Status: measured; rejected",
        "Scope: bounded CPU symbolic baseline; no checkpoint or ship claim",
        "",
        "## Decision",
        "",
        "The versioned reserved target codec and compiler-membership boundary are retained",
        "default-off, but the model-visible token hypothesis is rejected on this corpus.",
        "The canonical symbolic question exposes state and legal-set identity without an",
        "edit-intent channel, so multiple different transformations are gold for the same",
        "visible input. Reserved tokens change choices but cannot resolve that ambiguity.",
        "",
        "## Matched result",
        "",
        "| Arm | Seed | Exact action | Operator ID | Result AST | False admissions |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm, runs in result["arms"].items():
        for run in runs:
            lines.append(
                f"| `{arm}` | {run['seed']} | {run['exact_action_accuracy']:.3f} | "
                f"{run['operator_id_accuracy']:.3f} | {run['result_ast_accuracy']:.3f} | "
                f"{run['false_legal_admissions']} |"
            )
    lines.extend(
        [
            "",
            "All arms use the same train/held-out decisions, seeds, optimizer steps,",
            "learning rate, and parameter count. The treatment arms use explicit",
            "`<|openui_operator:v1|>` framing and typed canonical operator arguments.",
            "",
            "## Causal choice changes",
            "",
            "| Treatment | Seed | Changed | Rate | Correct | Wrong |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for arm, values in result["causal_changes_vs_result_ast_only"].items():
        for seed, item in zip(result["seeds"], values, strict=True):
            lines.append(
                f"| `{arm}` | {seed} | {item['changed']}/{item['eligible']} | "
                f"{item['change_rate']:.3f} | {item['correct_changes']} | "
                f"{item['wrong_changes']} |"
            )
    lines.extend(
        [
            "",
            "## Acceptance and honesty",
            "",
        ]
    )
    for key, value in result["acceptance"].items():
        if isinstance(value, bool):
            lines.append(f"- `{key}`: **{'pass' if value else 'fail'}**")
    lines.extend(
        [
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
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    args = parser.parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = build_version_stamp(
        "dsl.operators.reserved_tokens",
        "dsl.operators.legal_set",
        "harness.experiments.reserved_operator_baseline",
        "evals.cap2_operator",
    )
    manifest = json.loads(CAP2_MANIFEST.read_text(encoding="utf-8"))
    held_out_ids = set(manifest["source_record_ids"])
    train_records = [
        record
        for record in load_jsonl(TRAIN_SOURCE)
        if record.target_kind == "document"
    ][:4]
    held_out_records = [
        record
        for record in load_jsonl(HELD_OUT_SOURCE)
        if record.id in held_out_ids
    ]
    started = time.perf_counter()
    peak_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    with tempfile.TemporaryDirectory(prefix="e803-reserved-operator-") as temp:
        temp_root = Path(temp)
        train_corpus = build_symbolic_operator_corpus(
            records=train_records,
            output_dir=temp_root / "train",
            version="E803-train",
            version_stamp=stamp,
            config=OperatorCorpusConfig(
                max_roots=4,
                actions_per_state=4,
                max_combinations_per_operator=32,
                sibling_forks=False,
            ),
        )
        held_out_corpus = build_symbolic_operator_corpus(
            records=held_out_records,
            output_dir=temp_root / "held-out",
            version="E803-held-out",
            version_stamp=stamp,
            config=OperatorCorpusConfig(
                max_roots=2,
                actions_per_state=2,
                max_combinations_per_operator=32,
                sibling_forks=False,
            ),
        )
        result = run_reserved_operator_baseline(
            train_rows=_jsonl(Path(train_corpus["records_path"])),
            held_out_rows=_jsonl(Path(held_out_corpus["records_path"])),
            steps=args.steps,
            learning_rate=args.learning_rate,
        )
    peak_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_memory = max(peak_before, peak_after) * 1024
    elapsed = time.perf_counter() - started
    agentv = publish_agentv_evaluation(
        output_dir,
        name="e803-reserved-operator-baseline",
        claim="bounded_symbolic_baseline_rejected_not_ship",
        cases=_evidence_cases(result),
    )
    _rewrite_agentv_paths(output_dir)
    report = {
        "schema": "reserved_operator_baseline_report/v1",
        "run": {
            "experiment_id": "E803",
            "device": "cpu",
            "backend": "hashed_token_scorer",
            "max_wall_minutes": 3,
            "elapsed_seconds": elapsed,
            "peak_memory_bytes": peak_memory,
            "checkpoint": None,
            "ship_claim": False,
        },
        "corpora": {
            "train": {
                "root_count": train_corpus["root_count"],
                "record_count": train_corpus["record_count"],
                "content_fingerprint": train_corpus["content_fingerprint"],
            },
            "held_out": {
                "root_count": held_out_corpus["root_count"],
                "record_count": held_out_corpus["record_count"],
                "content_fingerprint": held_out_corpus["content_fingerprint"],
            },
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
                "experiment_id": "E803",
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
