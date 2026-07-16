"""Decode canvas feasibility vs ship gates and gold program lengths."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from slm_training.dsl.schema import ExampleRecord
from slm_training.harnesses.model_build.ship_gates import DEFAULT_SHIP_GATES
from slm_training.models.tokenizer import tokenize_text


def gold_token_len(openui: str, *, with_special: bool = True) -> int:
    """Token count for a gold OpenUI program (optional BOS/EOS)."""
    n = len(tokenize_text(openui))
    return n + (2 if with_special else 0)


def load_suite_gold_lengths(
    test_dir: Path,
    suite: str,
    *,
    rico_limit: int | None = None,
) -> list[tuple[str, int]]:
    """Return (record_id, gold_token_len) for a suite."""
    from slm_training.harnesses.model_build.data import load_suite_records

    records = load_suite_records(test_dir, suite)
    if suite == "rico_held" and rico_limit is not None:
        records = records[: max(0, int(rico_limit))]
    return [(r.id, gold_token_len(r.openui)) for r in records]


def max_achievable_parse_rate(
    gold_lengths: list[int],
    canvas_cap: int,
) -> float:
    """Upper bound on meaningful-program rate when targets exceed the cap."""
    if not gold_lengths:
        return 1.0
    fit = sum(1 for n in gold_lengths if n <= canvas_cap)
    return fit / len(gold_lengths)


def evaluate_decode_feasibility(
    test_dir: Path,
    *,
    canvas_cap: int,
    thresholds: dict[str, dict[str, float]] | None = None,
    rico_limit: int | None = None,
) -> dict[str, Any]:
    """
    Check whether the decode canvas can satisfy meaningful-program gates per suite.

    Returns per-suite max achievable parse and config-level pass/fail.
    """
    policy = thresholds or DEFAULT_SHIP_GATES
    suites: dict[str, Any] = {}
    failures: list[str] = []
    for suite_name, mins in policy.items():
        parse_gate = mins.get("meaningful_program_rate", mins.get("parse_rate"))
        if parse_gate is None:
            continue
        try:
            lengths = load_suite_gold_lengths(
                test_dir, suite_name, rico_limit=rico_limit
            )
        except (FileNotFoundError, ValueError):
            lengths = []
        max_parse = max_achievable_parse_rate(
            [n for _, n in lengths],
            canvas_cap,
        )
        longest = max((n for _, n in lengths), default=0)
        ok = max_parse >= float(parse_gate)
        suites[suite_name] = {
            "n": len(lengths),
            "canvas_cap": canvas_cap,
            "longest_gold_tokens": longest,
            "max_achievable_parse_rate": round(max_parse, 4),
            "parse_gate": float(parse_gate),
            "feasible": ok,
        }
        if not ok:
            failures.append(
                f"{suite_name}:max_parse={max_parse:.4f} need>={parse_gate} "
                f"(cap={canvas_cap}, longest_gold={longest})"
            )
    return {
        "canvas_cap": canvas_cap,
        "suites": suites,
        "failures": failures,
        "pass": len(failures) == 0,
        "note": (
            "Programs longer than the LTR canvas cap cannot reach meaningful parse. "
            "Raise grammar_ltr_max_tokens before training or eval."
        ),
    }


def classify_parse_failure(
    pred: str,
    *,
    error: str | None,
    gold: ExampleRecord | None = None,
    canvas_cap: int | None = None,
) -> str:
    """Map a failed meaningful-parse check to a diagnostic bucket."""
    if error and error.startswith("low_component_recall"):
        return "low_component_recall"
    if error == "no_placeholders":
        return "no_placeholders"
    if error in {"empty_root_stack", "empty_card", "no_content_components"}:
        return "trivial_layout"
    if error and (
        "parse" in error.lower()
        or "validation failed" in error.lower()
        or "unexpected token" in error.lower()
        or error.startswith("{")
    ):
        return "parse_error"
    if canvas_cap is not None and canvas_cap > 0:
        pred_len = gold_token_len(pred, with_special=True)
        if pred_len >= canvas_cap - 1 and error:
            return "truncated"
    if error:
        return "other"
    return "unknown"
