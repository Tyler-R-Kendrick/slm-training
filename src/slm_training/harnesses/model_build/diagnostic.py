"""Metric-ceiling and vocab-coverage diagnostics for train/eval alignment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slm_training.dsl.placeholders import extract_placeholders
from slm_training.dsl.schema import ExampleRecord, load_jsonl
from slm_training.harnesses.model_build.data import load_suite_records
from slm_training.harnesses.model_build.eval_runner import (
    _is_meaningful_program,
    _placeholder_fidelity,
    _placeholder_validity,
    component_type_recall,
    structural_similarity,
)
from slm_training.models.tokenizer import OpenUITokenizer, tokenize_text


@dataclass
class VocabCoverageReport:
    train_records: int
    vocab_size: int
    suites: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_records": self.train_records,
            "vocab_size": self.vocab_size,
            "suites": self.suites,
        }


def build_train_tokenizer(train_dir: Path) -> OpenUITokenizer:
    records = load_jsonl(train_dir / "records.jsonl")
    texts = [r.prompt for r in records] + [r.openui for r in records]
    return OpenUITokenizer.build(texts)


def _token_coverage(tokens: list[str], vocab: set[str]) -> dict[str, Any]:
    if not tokens:
        return {"n": 0, "missing": 0, "coverage": 1.0, "missing_samples": []}
    missing = [t for t in tokens if t not in vocab]
    return {
        "n": len(tokens),
        "missing": len(missing),
        "coverage": 1.0 - (len(missing) / len(tokens)),
        "missing_samples": sorted(set(missing))[:12],
    }


def vocab_coverage_report(
    train_dir: Path,
    test_dir: Path,
    *,
    suites: tuple[str, ...] = ("smoke", "held_out", "adversarial", "ood", "rico_held"),
) -> VocabCoverageReport:
    tokenizer = build_train_tokenizer(train_dir)
    vocab = set(tokenizer.token_to_id)
    train_records = load_jsonl(train_dir / "records.jsonl")
    suite_reports: dict[str, dict[str, Any]] = {}

    for suite in suites:
        try:
            records = load_suite_records(test_dir, suite)
        except FileNotFoundError:
            continue
        openui_tokens: list[str] = []
        placeholder_tokens: list[str] = []
        for record in records:
            openui_tokens.extend(tokenize_text(record.openui))
            for ph in record.placeholders or extract_placeholders(record.openui):
                placeholder_tokens.extend(tokenize_text(f'"{ph}"'))
        suite_reports[suite] = {
            "n": len(records),
            "openui_tokens": _token_coverage(openui_tokens, vocab),
            "placeholder_tokens": _token_coverage(placeholder_tokens, vocab),
        }

    return VocabCoverageReport(
        train_records=len(train_records),
        vocab_size=tokenizer.vocab_size,
        suites=suite_reports,
    )


def score_gold_as_prediction(record: ExampleRecord) -> dict[str, Any]:
    """Score gold openui as if it were the model prediction (metric ceiling)."""
    pred = record.openui
    ok, error, serialized = _is_meaningful_program(pred, gold=record)
    scored = serialized or pred
    return {
        "id": record.id,
        "parse_ok": ok,
        "error": error,
        "placeholder_fidelity": _placeholder_fidelity(scored, record),
        "placeholder_validity": _placeholder_validity(scored, record),
        "structural_similarity": structural_similarity(scored, record.openui),
        "component_type_recall": component_type_recall(scored, record.openui),
    }


def ceiling_report(
    test_dir: Path,
    *,
    suites: tuple[str, ...] = ("smoke", "held_out", "adversarial", "ood"),
) -> dict[str, Any]:
    """Aggregate gold-as-prediction scores per suite (should be ~1.0)."""
    out: dict[str, Any] = {}
    for suite in suites:
        try:
            records = load_suite_records(test_dir, suite)
        except FileNotFoundError:
            continue
        rows = [score_gold_as_prediction(r) for r in records]
        n = len(rows) or 1
        out[suite] = {
            "n": len(rows),
            "parse_rate": sum(1 for r in rows if r["parse_ok"]) / n,
            "placeholder_fidelity": sum(r["placeholder_fidelity"] for r in rows) / n,
            "placeholder_validity": sum(r["placeholder_validity"] for r in rows) / n,
            "structural_similarity": sum(r["structural_similarity"] for r in rows) / n,
            "component_type_recall": sum(r["component_type_recall"] for r in rows) / n,
            "failures": [r for r in rows if not r["parse_ok"] or r["placeholder_fidelity"] < 1.0],
        }
    return out


def run_full_diagnostic(
    train_dir: Path,
    test_dir: Path,
) -> dict[str, Any]:
    return {
        "vocab_coverage": vocab_coverage_report(train_dir, test_dir).to_dict(),
        "ceiling": ceiling_report(test_dir),
    }
