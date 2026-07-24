#!/usr/bin/env python3
"""Reverify archived failures locally without regenerating model outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import quote

from slm_training.data.verify.runtime import run_preview_verifier
from slm_training.evals.agentv import publish_agentv_evaluation
from slm_training.evals.oracle_scoring_replay import score_prediction
from slm_training.evals.verifier_cascade import default_openui_cascade
from slm_training.harnesses.eval.harness_replay import (
    REPLAY_SUITES,
    UNKNOWN_NOT_CAPTURED,
    ArchivedFailureV1,
    collect_archived_failures,
    harness_provenance_id,
    prediction_lineage,
    select_suite_stratified_failures,
)
from slm_training.harnesses.model_build.data import load_suite_records
from slm_training.harnesses.model_build.decode_feasibility import (
    evaluate_decode_feasibility,
    gold_token_len,
)
from slm_training.levers import INTERRUPT_AFTER_SECONDS, MAX_RUN_MINUTES
from slm_training.versioning import build_version_stamp

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_DATA = (
    ROOT / "src/slm_training/resources/data/eval/e763_symbol_only_eval_r2_20260722"
)


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
    raw_prefix = str(output_dir.resolve())
    encoded_prefix = quote(raw_prefix, safe="")
    for path in (output_dir / "agentv").rglob("*"):
        if not path.is_file() or path.suffix not in {".json", ".jsonl"}:
            continue
        text = path.read_text(encoding="utf-8")
        text = text.replace(raw_prefix, "output-dir://")
        text = text.replace(encoded_prefix, quote("output-dir://", safe=""))
        path.write_text(text, encoding="utf-8")


def _redacted_score(detail: dict[str, Any]) -> dict[str, Any]:
    """Keep scorer facts, never the archived prediction or its serialization."""
    return {
        key: detail.get(key)
        for key in (
            "parse_ok",
            "meaningful_program_v1",
            "binding_aware_meaningful_v2",
            "syntax_parse_valid",
            "raw_syntax_valid",
            "error",
            "placeholder_fidelity",
            "component_type_recall",
            "reward_score",
            "prediction_sha256",
            "source_record_sha256",
        )
    }


def _runtime_result(source: str, *, timeout_s: float) -> dict[str, Any]:
    try:
        evidence = run_preview_verifier(source, timeout_s=timeout_s)
    except (TimeoutError, subprocess.TimeoutExpired):
        return {"status": "timeout", "rendered": None}
    except RuntimeError as exc:
        return {"status": "error", "rendered": None, "error_kind": type(exc).__name__}
    return {
        "status": "pass" if evidence.rendered else "fail",
        "rendered": evidence.rendered,
        "console_error_count": len(evidence.console_errors),
        "behavior_error_count": len(evidence.behavior_errors),
        "interaction_count": len(evidence.interaction_trace),
    }


def _reverify_case(
    case: ArchivedFailureV1,
    record: Any,
    *,
    runtime_timeout_s: float,
    target_lengths: tuple[int, ...],
) -> dict[str, Any]:
    """Score one immutable archived output through existing verifier owners."""
    detail = score_prediction(record, case.raw_prediction)
    cascade = default_openui_cascade().evaluate(case.event_id, case.raw_prediction)
    token_length = gold_token_len(case.raw_prediction)
    runtime = (
        _runtime_result(case.raw_prediction, timeout_s=runtime_timeout_s)
        if detail["raw_syntax_valid"]
        else {"status": "not_applicable", "rendered": None}
    )
    classifications = ["stable_failure"]
    if any(token_length >= target for target in target_lengths):
        classifications.append("truncation_sensitive")
    if case.provenance.canvas_cap is not None and token_length >= case.provenance.canvas_cap:
        classifications.append("canvas_sensitive")
    if runtime["status"] == "timeout":
        classifications.append("runtime_timeout_sensitive")
    if runtime["status"] == "error":
        classifications.append("non_reproducible")
    lineage = prediction_lineage(case.raw_prediction)
    return {
        "event_id": case.event_id,
        "suite": case.suite,
        "record_id": case.record_id,
        "original_failure": case.original_failure,
        **lineage,
        "raw_prediction_preserved": (
            lineage["raw_prediction_sha256"] == case.raw_prediction_sha256
        ),
        "harness_provenance_id": harness_provenance_id(case.provenance),
        "source_eval_sha256": case.provenance.source_eval_sha256,
        "source_token_length": token_length,
        "target_length_feasibility": {
            str(target): token_length < target for target in target_lengths
        },
        "canonical_score": _redacted_score(detail),
        "parser_fallback": {
            "owner": "default_openui_cascade",
            "final_status": cascade.final_status.value,
            "prune_stage_id": cascade.prune_stage_id,
        },
        "runtime_verifier": runtime,
        "repair": {"status": UNKNOWN_NOT_CAPTURED, "label": None},
        "decoder_timeout": {"status": UNKNOWN_NOT_CAPTURED, "label": None},
        "decoder_fallback": {"status": UNKNOWN_NOT_CAPTURED, "label": None},
        "classifications": classifications,
        "actual_decode_replayed": False,
    }


def _records_by_suite(test_dir: Path) -> dict[str, dict[str, Any]]:
    return {
        suite: {record.id: record for record in load_suite_records(test_dir, suite)}
        for suite in REPLAY_SUITES
    }


def _partial_decoder_trace_count(
    archive_root: Path, cases: list[ArchivedFailureV1]
) -> int:
    """Count legacy prefix traces that link to a selected archived output.

    These traces lack complete per-step provenance, so the count is diagnostic
    only and is deliberately excluded from the causal-flip denominator.
    """
    selected: dict[str, set[str]] = {}
    for case in cases:
        source, _ = case.event_id.rsplit("#", 1)
        selected.setdefault(source, set()).add(case.record_id)
    linked = 0
    for source, record_ids in selected.items():
        payload = json.loads((archive_root / source).read_text(encoding="utf-8"))
        details = payload.get("details") or []
        traces = (payload.get("decode_stats") or {}).get("constrained_selection_traces") or []
        groups: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        last_position: int | None = None
        for trace in traces:
            if not isinstance(trace, dict):
                continue
            position = trace.get("position")
            if current and isinstance(position, int) and last_position is not None and position <= last_position:
                groups.append(current)
                current = []
            current.append(trace)
            last_position = position if isinstance(position, int) else None
        if current:
            groups.append(current)
        trace_record_ids: set[str] = set()
        for group in groups:
            prefix = max(
                (
                    trace.get("prefix_text", "")
                    for trace in group
                    if isinstance(trace.get("prefix_text"), str)
                ),
                key=len,
                default="",
            )
            matches = [
                detail.get("id")
                for detail in details
                if isinstance(detail, dict)
                and isinstance(detail.get("prediction"), str)
                and detail["prediction"].startswith(prefix)
            ]
            if len(matches) == 1 and isinstance(matches[0], str):
                trace_record_ids.add(matches[0])
        linked += len(trace_record_ids & record_ids)
    return linked


def _agentv_cases(payload: dict[str, Any]) -> list[dict[str, Any]]:
    summary = payload["summary"]
    return [
        {
            "id": "byte-preserving-reverification",
            "criteria": "Replay exactly 100 suite-stratified archived failures without regenerating a model output.",
            "pass": summary["meets_required_n"] and summary["all_raw_bytes_preserved"],
            "checks": {
                "n_is_100": summary["n"] == 100,
                "suite_stratified": all(
                    n == 20 for n in summary["selected_n_by_suite"].values()
                ),
                "raw_bytes_preserved": summary["all_raw_bytes_preserved"],
                "no_model_output_regenerated": summary["actual_decode_replayed"] is False,
            },
            "result": summary,
            "metadata": {"honesty": "byte_preserving_reverification_not_ship"},
        },
        {
            "id": "decoder-causal-boundary",
            "criteria": "Do not fabricate decoder timeout, fallback, or repair label flips from archived output bytes.",
            "pass": summary["decoder_causal_flip_rate"] is None
            and summary["architecture_claims_blocked"],
            "checks": {
                "decoder_causal_flip_is_unknown": summary["decoder_causal_flip_rate"] is None,
                "architecture_claims_blocked": summary["architecture_claims_blocked"],
            },
            "result": {
                "decoder_trace_linked_n": summary["decoder_trace_linked_n"],
                "partial_decoder_trace_linked_n": summary[
                    "partial_decoder_trace_linked_n"
                ],
                "required_n": summary["required_n"],
            },
            "metadata": {"honesty": "missing_decoder_traces_fail_closed"},
        },
    ]


def build_replay_report(
    archive_root: Path,
    test_dir: Path,
    *,
    suite_quota: int = 20,
    runtime_timeout_s: float = 3.0,
    target_lengths: tuple[int, ...] = (64, 128, 256),
) -> dict[str, Any]:
    """Build a redacted historical byte-reverification matrix."""
    started = time.monotonic()
    cases = select_suite_stratified_failures(
        collect_archived_failures(archive_root, limit=None), suite_quota=suite_quota
    )
    records = _records_by_suite(test_dir)
    rows = []
    for case in cases:
        if time.monotonic() - started >= INTERRUPT_AFTER_SECONDS:
            raise TimeoutError("replay exceeded the canonical interrupt budget")
        record = records[case.suite].get(case.record_id)
        if record is None:
            raise ValueError(f"missing canonical record {case.suite}/{case.record_id}")
        rows.append(
            _reverify_case(
                case,
                record,
                runtime_timeout_s=runtime_timeout_s,
                target_lengths=target_lengths,
            )
        )
    selected_n_by_suite = Counter(row["suite"] for row in rows)
    current_passes = sum(
        bool(row["canonical_score"]["meaningful_program_v1"]) for row in rows
    )
    partial_decoder_trace_linked_n = _partial_decoder_trace_count(archive_root, cases)
    decoder_trace_linked_n = 0  # No legacy trace has complete causal provenance.
    required_n = suite_quota * len(REPLAY_SUITES)
    summary = {
        "n": len(rows),
        "required_n": required_n,
        "meets_required_n": len(rows) == required_n,
        "selected_n_by_suite": {
            suite: selected_n_by_suite.get(suite, 0) for suite in REPLAY_SUITES
        },
        "all_raw_bytes_preserved": all(row["raw_prediction_preserved"] for row in rows),
        "actual_decode_replayed": False,
        "byte_reverification_flip_rate": current_passes / len(rows) if rows else None,
        "labels_flip_rate": current_passes / len(rows) if rows else None,
        "decoder_causal_flip_rate": None,
        "decoder_trace_linked_n": decoder_trace_linked_n,
        "partial_decoder_trace_linked_n": partial_decoder_trace_linked_n,
        "architecture_claims_blocked": (
            (current_passes / len(rows) >= 0.15 if rows else True)
            or decoder_trace_linked_n < required_n
        ),
        "architecture_block_reasons": [
            "decoder_causal_coverage_below_required_n"
        ],
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    canvas_feasibility = [
        evaluate_decode_feasibility(test_dir, canvas_cap=cap)
        for cap in target_lengths
    ]
    return {
        "schema": "HarnessArtifactAuditV2",
        "claim_class": "authorized_local_byte_reverification_not_ship",
        "matrix_status": "completed_byte_reverification_decoder_causal_unmeasured",
        "run": {
            "device": "cpu",
            "steps": 0,
            "backend": "local_archived_bytes",
            "matrix_set": "SLM-281/AP-003",
            "suite_n": summary["selected_n_by_suite"],
            "honesty": "raw_bytes_preserved_no_model_regeneration",
            "max_run_minutes": MAX_RUN_MINUTES,
            "runtime_timeout_seconds": runtime_timeout_s,
        },
        "summary": summary,
        "arms": {
            "primary": "production_oracle_score_prediction",
            "parser_fallback": "default_openui_cascade",
            "runtime": "openui_preview_chromium",
            "target_lengths": list(target_lengths),
            "canvas_feasibility": canvas_feasibility,
            "repair": UNKNOWN_NOT_CAPTURED,
            "decoder_timeout": UNKNOWN_NOT_CAPTURED,
            "decoder_fallback": UNKNOWN_NOT_CAPTURED,
        },
        "rows": rows,
        "version_stamp": build_version_stamp("harness.eval.replay"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--eval-data-root", type=Path, default=DEFAULT_EVAL_DATA)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--suite-quota", type=int, default=20)
    parser.add_argument("--runtime-timeout-s", type=float, default=3.0)
    parser.add_argument("--target-lengths", default="64,128,256")
    parser.add_argument("--no-publish-agentv", action="store_true")
    parser.add_argument(
        "--allow-design-output",
        action="store_true",
        help="acknowledge that a redacted local re-verification summary is approved for docs/design publication",
    )
    args = parser.parse_args(argv)
    design_root = (ROOT / "docs" / "design").resolve()
    if args.output.resolve().is_relative_to(design_root) and not args.allow_design_output:
        parser.error(
            "refusing to publish archive-derived evidence under docs/design; "
            "use --allow-design-output only for an authorized redacted summary"
        )
    target_lengths = tuple(
        sorted({int(value) for value in args.target_lengths.split(",") if value})
    )
    if not target_lengths or target_lengths[0] < 1:
        parser.error("--target-lengths must contain positive integers")
    try:
        payload = build_replay_report(
            args.archive_root,
            args.eval_data_root,
            suite_quota=args.suite_quota,
            runtime_timeout_s=args.runtime_timeout_s,
            target_lengths=target_lengths,
        )
    except (TimeoutError, ValueError) as exc:
        parser.error(str(exc))
    run_dir = (args.run_dir or args.output.parent / "slm281-harness-replay").resolve()
    if not args.no_publish_agentv:
        agentv = publish_agentv_evaluation(
            run_dir,
            name="slm281-harness-byte-reverification",
            claim="byte_preserving_harness_reverification_not_ship",
            cases=_agentv_cases(payload),
            version_stamp=payload["version_stamp"],
        )
        _rewrite_agentv_paths(run_dir)
        payload["agentv"] = _portable(agentv, run_dir)
        spec_path = run_dir / "agentv/slm281-harness-byte-reverification.eval.jsonl"
        payload["agentv"]["spec_sha256"] = hashlib.sha256(spec_path.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"n": payload["summary"]["n"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
