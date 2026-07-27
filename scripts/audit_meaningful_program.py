"""Replay v1/v2 meaningful-program metrics over durable generation envelopes."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from slm_training.levers import DEFAULT_EVAL_DATA_DIR
from typing import Any

from slm_training.dsl.schema import ExampleRecord, load_jsonl
from slm_training.data.contract import GenerationRequest
from slm_training.evals.agentv import publish_agentv_evaluation
from slm_training.evals.meaningful_program import (
    aggregate_meaning_reports_v2,
    binding_aware_meaningful_v2,
)
from slm_training.harnesses.model_build.data import load_suite_records
from slm_training.harnesses.model_build.eval_runner import meaningful_program_v1


def _matrix(pairs: list[tuple[bool, bool]]) -> dict[str, Any]:
    counts = Counter(f"v1_{str(a).lower()}_v2_{str(b).lower()}" for a, b in pairs)
    n = len(pairs)
    agree = sum(a == b for a, b in pairs) / n if n else None
    if not n:
        kappa = None
    else:
        v1_pos = sum(a for a, _ in pairs) / n
        v2_pos = sum(b for _, b in pairs) / n
        expected = v1_pos * v2_pos + (1 - v1_pos) * (1 - v2_pos)
        kappa = (agree - expected) / (1 - expected) if expected < 1 else None
    return {"n": n, "counts": dict(sorted(counts.items())), "agreement": agree, "kappa": kappa}


def _labeled_metrics(pairs: list[tuple[bool, bool]]) -> dict[str, Any]:
    if not pairs:
        return {
            "status": "UNKNOWN",
            "n": 0,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "precision": None,
            "recall": None,
            "accuracy": None,
            "kappa": None,
        }
    tp = sum(pred and gold for pred, gold in pairs)
    tn = sum(not pred and not gold for pred, gold in pairs)
    fp = sum(pred and not gold for pred, gold in pairs)
    fn = sum(not pred and gold for pred, gold in pairs)
    n = len(pairs)
    accuracy = (tp + tn) / n
    pred_pos = (tp + fp) / n
    gold_pos = (tp + fn) / n
    expected = pred_pos * gold_pos + (1 - pred_pos) * (1 - gold_pos)
    return {
        "status": "available",
        "n": n,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "accuracy": accuracy,
        "kappa": (accuracy - expected) / (1 - expected) if expected < 1 else None,
    }


def _confusion(pairs: list[tuple[bool, bool]]) -> dict[str, Any]:
    """Treat v1 as the historical reference and v2 as the candidate prediction."""
    tp = sum(v1 and v2 for v1, v2 in pairs)
    tn = sum(not v1 and not v2 for v1, v2 in pairs)
    fp = sum(not v1 and v2 for v1, v2 in pairs)
    fn = sum(v1 and not v2 for v1, v2 in pairs)
    n = len(pairs)
    accuracy = (tp + tn) / n if n else None
    if n:
        v1_pos = (tp + fn) / n
        v2_pos = (tp + fp) / n
        expected = v1_pos * v2_pos + (1 - v1_pos) * (1 - v2_pos)
        kappa = (accuracy - expected) / (1 - expected) if expected < 1 else None
    else:
        kappa = None
    return {
        "n": n,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "accuracy": accuracy,
        "cohen_kappa": kappa,
    }


def audit(
    eval_paths: list[Path],
    record_paths: list[Path],
    label_paths: list[Path] | None = None,
) -> dict[str, Any]:
    """Replay explicit eval/record envelopes; incomplete legacy text stays UNKNOWN."""
    records = {record.id: record for path in record_paths for record in load_jsonl(path)}
    labels: dict[str, dict[str, bool]] = {}
    for path in label_paths or ():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            labels.setdefault(str(row["id"]), {}).update(
                {
                    str(key): value
                    for key, value in (row.get("labels") or {}).items()
                    if isinstance(value, bool)
                }
            )
    rows = []
    reports = []
    comparison: list[tuple[bool, bool]] = []
    label_pairs: dict[str, list[tuple[bool, bool]]] = {}
    unknown_reasons: Counter[str] = Counter()
    for path in eval_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for detail in payload.get("details") or ():
            case_id = str(detail.get("id") or "")
            record = records.get(case_id)
            pred = detail.get("prediction")
            truncated = (
                isinstance(pred, str)
                and len(pred) == 500
                and not detail.get("prediction_sha256")
            )
            digest_mismatch = (
                isinstance(pred, str)
                and detail.get("prediction_sha256") is not None
                and detail["prediction_sha256"]
                != hashlib.sha256(pred.encode()).hexdigest()
            )
            if (
                record is None
                or not isinstance(pred, str)
                or truncated
                or digest_mismatch
            ):
                reason = (
                    "stored_prediction_may_be_truncated"
                    if truncated
                    else (
                        "prediction_digest_mismatch"
                        if digest_mismatch
                        else "generation_or_record_missing"
                    )
                )
                unknown_reasons[reason] += 1
                rows.append(
                    {
                        "id": case_id,
                        "prediction_status": "UNKNOWN",
                        "meaningful_program_v1": {"verdict": None},
                        "binding_aware_meaningful_v2": {
                            "verdict": False,
                            "reason_codes": [reason],
                        },
                    }
                )
                continue
            v1, reason, _serialized = meaningful_program_v1(pred, gold=record)
            v2 = binding_aware_meaningful_v2(
                pred, record=record, request=_effective_request(record, payload)
            )
            reports.append(v2)
            comparison.append((v1, v2.verdict))
            explicit = dict(labels.get(case_id) or {})
            for key, value in detail.items():
                if key.endswith("_label") and isinstance(value, bool):
                    explicit[key.removesuffix("_label")] = value
            for name, value in explicit.items():
                label_pairs.setdefault(name, []).append((v2.verdict, value))
            rows.append(
                {
                    "id": case_id,
                    "prediction_status": "complete",
                    "meaningful_program_v1": {"verdict": v1, "reason": reason},
                    "binding_aware_meaningful_v2": v2.to_dict(),
                }
            )
    summary = aggregate_meaning_reports_v2(reports)
    total = len(rows)
    covered = sum(report.coverage_known for report in reports)
    summary["strict_rate"] = sum(report.verdict for report in reports) / total if total else 0.0
    summary["n"] = total
    summary["replayable_n"] = len(reports)
    summary["coverage"] = covered / total if total else 0.0
    reasons = Counter(summary["reason_prevalence"])
    reasons.update(unknown_reasons)
    summary["reason_prevalence"] = dict(sorted(reasons.items()))
    return {
        "n": total,
        "rows": rows,
        "v2": summary,
        "v1_v2_confusion": _confusion(comparison),
        "label_comparisons": {
            name: {"available_n": len(pairs), **_labeled_metrics(pairs)}
            for name, pairs in sorted(label_pairs.items())
        },
    }


def _eval_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.glob("eval_*.json"))


def _available_labels(detail: dict[str, Any]) -> dict[str, bool]:
    labels = detail.get("semantic_labels")
    if not isinstance(labels, dict):
        return {}
    result: dict[str, bool] = {}
    for label in ("independent_judge", "human", "agentv", "efs0_04"):
        value = labels.get(label)
        if (
            isinstance(value, dict)
            and isinstance(value.get("verdict"), bool)
            and isinstance(value.get("provenance"), str)
            and value["provenance"].strip()
        ):
            result[label] = value["verdict"]
    return result


def _effective_request(
    record: ExampleRecord, payload: dict[str, Any]
) -> GenerationRequest:
    request = GenerationRequest.from_record(record, include_design_md=False)
    policy = payload.get("evaluation_policy")
    if not (
        isinstance(policy, dict) and policy.get("slot_contract_in_context") is True
    ):
        request = GenerationRequest.from_dict({**request.to_dict(), "slot_contract": []})
    return request


def _verified_checkpoint(
    payload: dict[str, Any], eval_path: Path
) -> tuple[str | None, dict[str, Any], str | None]:
    expected = payload.get("checkpoint_sha256")
    raw_path = payload.get("checkpoint")
    if not (
        isinstance(expected, str)
        and len(expected) == 64
        and all(char in "0123456789abcdef" for char in expected)
    ):
        return None, {"status": "UNKNOWN"}, "checkpoint_sha256_missing_or_invalid"
    if not isinstance(raw_path, str) or not raw_path:
        return None, {"status": "UNKNOWN"}, "checkpoint_path_missing"
    checkpoint = Path(raw_path)
    candidates = [checkpoint] if checkpoint.is_absolute() else [
        Path.cwd() / checkpoint,
        *(parent / checkpoint for parent in eval_path.parents),
    ]
    resolved = next((candidate for candidate in candidates if candidate.is_file()), None)
    if resolved is None:
        return (
            None,
            {"status": "UNKNOWN", "path": raw_path, "expected_sha256": expected},
            "checkpoint_file_missing",
        )
    actual = hashlib.sha256(resolved.read_bytes()).hexdigest()
    verification = {
        "status": "PASS" if actual == expected else "FAIL",
        "path": str(resolved),
        "expected_sha256": expected,
        "actual_sha256": actual,
    }
    if actual != expected:
        return None, verification, "checkpoint_digest_mismatch"
    return expected, verification, None


def _frontier_set(label: str, path: Path, test_dir: Path) -> dict[str, Any]:
    reports = []
    comparisons: list[tuple[bool, bool]] = []
    labels: dict[str, list[tuple[bool, bool, bool]]] = {
        "independent_judge": [],
        "human": [],
        "agentv": [],
        "efs0_04": [],
    }
    reasons: Counter[str] = Counter()
    failures: list[str] = []
    cases: list[dict[str, Any]] = []
    suite_counts: Counter[str] = Counter()
    checkpoint_sha256s: set[str] = set()
    checkpoint_verifications: list[dict[str, Any]] = []
    for eval_path in _eval_files(path):
        payload = json.loads(eval_path.read_text(encoding="utf-8"))
        details = payload.get("details")
        if not isinstance(details, list):
            failures.append(f"{eval_path.name}:details_missing_or_invalid")
            details = []
        declared_n = payload.get("n")
        if not isinstance(declared_n, int) or declared_n != len(details):
            failures.append(
                f"{eval_path.name}:declared_n_mismatch:"
                f"{declared_n!r}!={len(details)}"
            )
        case_ids = [str(detail.get("id") or "") for detail in details]
        if any(not case_id for case_id in case_ids):
            failures.append(f"{eval_path.name}:case_id_missing")
        if len(set(case_ids)) != len(case_ids):
            failures.append(f"{eval_path.name}:duplicate_case_id")
        checkpoint_sha256, verification, checkpoint_failure = _verified_checkpoint(
            payload, eval_path
        )
        checkpoint_verifications.append(
            {"eval_json": str(eval_path), **verification}
        )
        if checkpoint_failure is not None:
            failures.append(f"{eval_path.name}:{checkpoint_failure}")
        elif checkpoint_sha256 is not None:
            checkpoint_sha256s.add(checkpoint_sha256)
        suite = str(payload.get("suite") or eval_path.stem.removeprefix("eval_"))
        try:
            records = {record.id: record for record in load_suite_records(test_dir, suite)}
        except (FileNotFoundError, ValueError) as exc:
            failures.append(f"{suite}:record_source_unavailable:{exc}")
            continue
        for detail in details:
            case_id = str(detail.get("id") or "")
            pred = detail.get("prediction")
            record = records.get(case_id)
            unknown_reason = None
            if record is None:
                unknown_reason = "test_record_missing"
            elif record.target_kind != "document":
                unknown_reason = "non_document_target"
            elif not isinstance(pred, str):
                unknown_reason = "prediction_missing"
            elif len(pred) == 500 and not detail.get("prediction_sha256"):
                unknown_reason = "stored_prediction_may_be_truncated"
            elif detail.get("prediction_sha256") is not None and detail[
                "prediction_sha256"
            ] != hashlib.sha256(pred.encode("utf-8")).hexdigest():
                unknown_reason = "prediction_digest_mismatch"
            if unknown_reason is not None:
                failures.append(f"{suite}:{case_id}:{unknown_reason}")
                reasons[unknown_reason] += 1
                cases.append(
                    {
                        "id": case_id,
                        "suite": suite,
                        "eval_json": str(eval_path),
                        "prediction_status": "UNKNOWN",
                        "meaningful_program_v1": {
                            "status": "UNKNOWN",
                            "verdict": None,
                            "reason_code": unknown_reason,
                        },
                        "binding_aware_meaningful_v2": {
                            "status": "UNKNOWN",
                            "verdict": False,
                            "reason_codes": [unknown_reason],
                        },
                        "labels": _available_labels(detail),
                    }
                )
                continue
            assert record is not None and isinstance(pred, str)
            request = _effective_request(record, payload)
            v1, _reason, _serialized = meaningful_program_v1(pred, gold=record)
            v2 = binding_aware_meaningful_v2(pred, record=record, request=request)
            replay = binding_aware_meaningful_v2(pred, record=record, request=request)
            if replay.to_dict() != v2.to_dict():
                failures.append(f"{suite}:{case_id}:nondeterministic_metric_replay")
                continue
            reports.append(v2)
            comparisons.append((v1, v2.verdict))
            reasons.update(v2.reason_codes)
            suite_counts[suite] += 1
            explicit_labels = _available_labels(detail)
            for key, value in explicit_labels.items():
                labels[key].append((v1, v2.verdict, value))
            cases.append(
                {
                    "id": case_id,
                    "suite": suite,
                    "eval_json": str(eval_path),
                    "prediction_status": "COMPLETE",
                    "meaningful_program_v1": {
                        "status": "PASS" if v1 else "FAIL",
                        "verdict": v1,
                        "reason_code": _reason,
                    },
                    "binding_aware_meaningful_v2": {
                        "status": (
                            "PASS"
                            if v2.verdict
                            else "FAIL" if v2.coverage_known else "UNKNOWN"
                        ),
                        **v2.to_dict(),
                    },
                    "effective_request": request.to_dict(),
                    "prediction_sha256": hashlib.sha256(
                        pred.encode("utf-8")
                    ).hexdigest(),
                    "source_record_sha256": hashlib.sha256(
                        json.dumps(
                            record.to_dict(), sort_keys=True, separators=(",", ":")
                        ).encode("utf-8")
                    ).hexdigest(),
                    "labels": explicit_labels,
                }
            )
    complete = bool(reports) and not failures
    base_v2 = aggregate_meaning_reports_v2(reports)
    covered = sum(report.coverage_known for report in reports)
    positives = sum(report.verdict for report in reports)
    total = len(cases)
    v2_summary = {
        **base_v2,
        "n": total,
        "replayable_n": len(reports),
        "covered_n": covered,
        "strict_rate": positives / total if total else 0.0,
        "coverage_conditioned_rate": positives / covered if covered else 0.0,
        "coverage": covered / total if total else 0.0,
        "reason_prevalence": dict(sorted(reasons.items())),
    }
    return {
        "label": label,
        "source": str(path),
        "checkpoint_sha256s": sorted(checkpoint_sha256s),
        "checkpoint_verifications": checkpoint_verifications,
        "replayable": complete,
        "failures": failures,
        "suite_counts": dict(sorted(suite_counts.items())),
        "v1_v2": _matrix(comparisons),
        "v2": v2_summary,
        "reason_prevalence": dict(sorted(reasons.items())),
        "external_labels": {
            key: {
                "status": "available" if value else "UNKNOWN",
                "n": len(value),
                "v1_vs_label": _labeled_metrics(
                    [(v1, gold) for v1, _v2, gold in value]
                ),
                "v2_vs_label": _labeled_metrics(
                    [(v2, gold) for _v1, v2, gold in value]
                ),
            }
            for key, value in labels.items()
        },
        "cases": cases,
    }


def _gaming_set(path: Path) -> dict[str, Any]:
    reports = []
    failures = []
    for line in path.read_text(encoding="utf-8").splitlines():
        case = json.loads(line)
        record = ExampleRecord(
            id=str(case["id"]),
            prompt=str(case["prompt"]),
            openui=str(case["prediction"]),
            split="adversarial",
            source="deterministic",
        )
        report = binding_aware_meaningful_v2(case["prediction"], record=record)
        replay = binding_aware_meaningful_v2(case["prediction"], record=record)
        reports.append(report)
        missing = sorted(set(case["expected_reason_codes"]) - set(report.reason_codes))
        if replay.to_dict() != report.to_dict():
            failures.append({"id": case["id"], "reason": "nondeterministic_metric_replay"})
        elif report.verdict is not case["expected_verdict"] or missing:
            failures.append(
                {"id": case["id"], "verdict": report.verdict, "missing_reasons": missing}
            )
    return {
        "label": "deterministic_gaming_corpus",
        "source": str(path),
        "replayable": not failures,
        "n": len(reports),
        "failures": failures,
        "v2": aggregate_meaning_reports_v2(reports),
    }


def _capture_replay_bundle(
    generation_sets: list[tuple[str, Path]], test_dir: Path
) -> dict[str, Any]:
    records: dict[str, dict[str, Any]] = {}
    sets: list[dict[str, Any]] = []
    for label, path in generation_sets:
        for eval_path in _eval_files(path):
            payload = json.loads(eval_path.read_text(encoding="utf-8"))
            suite = str(payload.get("suite") or eval_path.stem.removeprefix("eval_"))
            suite_records = {
                record.id: record for record in load_suite_records(test_dir, suite)
            }
            checkpoint_sha256, verification, failure = _verified_checkpoint(
                payload, eval_path
            )
            if failure is not None or checkpoint_sha256 is None:
                raise ValueError(f"{label}:{eval_path.name}:{failure}")
            details = payload.get("details")
            if not isinstance(details, list):
                raise ValueError(f"{label}:{eval_path.name}:details_missing_or_invalid")
            captured_details = []
            for detail in details:
                case_id = str(detail.get("id") or "")
                record = suite_records.get(case_id)
                if record is None:
                    raise ValueError(f"{label}:{suite}:{case_id}:test_record_missing")
                key = f"{suite}:{case_id}"
                record_payload = record.to_dict()
                prior = records.setdefault(key, record_payload)
                if prior != record_payload:
                    raise ValueError(f"{label}:{key}:record_content_changed")
                pred = detail.get("prediction")
                if not isinstance(pred, str):
                    raise ValueError(f"{label}:{suite}:{case_id}:prediction_missing")
                captured_details.append(
                    {
                        **detail,
                        "prediction_sha256": hashlib.sha256(
                            pred.encode("utf-8")
                        ).hexdigest(),
                        "source_record_sha256": hashlib.sha256(
                            json.dumps(
                                record_payload, sort_keys=True, separators=(",", ":")
                            ).encode("utf-8")
                        ).hexdigest(),
                        "generation_request": _effective_request(
                            record, payload
                        ).to_dict(),
                    }
                )
            sets.append(
                {
                    "label": label,
                    "suite": suite,
                    "declared_n": payload.get("n"),
                    "evaluation_policy": payload.get("evaluation_policy") or {},
                    "checkpoint": payload.get("checkpoint"),
                    "checkpoint_sha256": checkpoint_sha256,
                    "checkpoint_verification_at_capture": verification,
                    "source_eval_json": str(eval_path),
                    "source_eval_sha256": hashlib.sha256(eval_path.read_bytes()).hexdigest(),
                    "details": captured_details,
                }
            )
    return {
        "format": "binding_aware_meaningful_v2_replay_bundle",
        "version": 1,
        "records": records,
        "sets": sets,
    }


def _replay_bundle_set(
    row: dict[str, Any], records: dict[str, ExampleRecord]
) -> dict[str, Any]:
    label = str(row["label"])
    suite = str(row["suite"])
    details = row.get("details")
    failures: list[str] = []
    if not isinstance(details, list):
        details = []
        failures.append("details_missing_or_invalid")
    if row.get("declared_n") != len(details):
        failures.append(f"declared_n_mismatch:{row.get('declared_n')!r}!={len(details)}")
    ids = [str(detail.get("id") or "") for detail in details]
    if any(not case_id for case_id in ids):
        failures.append("case_id_missing")
    if len(ids) != len(set(ids)):
        failures.append("duplicate_case_id")
    checkpoint_sha256 = row.get("checkpoint_sha256")
    verification = row.get("checkpoint_verification_at_capture")
    if not (
        isinstance(checkpoint_sha256, str)
        and len(checkpoint_sha256) == 64
        and isinstance(verification, dict)
        and verification.get("status") == "PASS"
        and verification.get("actual_sha256") == checkpoint_sha256
    ):
        failures.append("captured_checkpoint_verification_invalid")
    reports = []
    comparisons: list[tuple[bool, bool]] = []
    reasons: Counter[str] = Counter()
    cases: list[dict[str, Any]] = []
    labels: dict[str, list[tuple[bool, bool, bool]]] = {
        "independent_judge": [],
        "human": [],
        "agentv": [],
        "efs0_04": [],
    }
    for detail in details:
        case_id = str(detail.get("id") or "")
        record = records.get(f"{suite}:{case_id}")
        pred = detail.get("prediction")
        request: GenerationRequest | None = None
        unknown_reason = None
        if record is None:
            unknown_reason = "test_record_missing"
        elif not isinstance(pred, str):
            unknown_reason = "prediction_missing"
        elif not isinstance(detail.get("prediction_sha256"), str):
            unknown_reason = "prediction_digest_missing"
        elif detail.get("prediction_sha256") is not None and detail[
            "prediction_sha256"
        ] != hashlib.sha256(pred.encode()).hexdigest():
            unknown_reason = "prediction_digest_mismatch"
        elif len(pred) == 500 and not detail.get("prediction_sha256"):
            unknown_reason = "stored_prediction_may_be_truncated"
        elif detail.get("source_record_sha256") != hashlib.sha256(
            json.dumps(
                record.to_dict(), sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest():
            unknown_reason = "source_record_digest_missing_or_mismatch"
        elif not isinstance(detail.get("generation_request"), dict):
            unknown_reason = "effective_generation_request_missing"
        else:
            try:
                request = GenerationRequest.from_dict(detail["generation_request"])
            except (TypeError, ValueError):
                unknown_reason = "effective_generation_request_invalid"
        if unknown_reason is not None:
            failures.append(f"{suite}:{case_id}:{unknown_reason}")
            reasons[unknown_reason] += 1
            continue
        assert record is not None and isinstance(pred, str) and request is not None
        v1, reason, _serialized = meaningful_program_v1(pred, gold=record)
        v2 = binding_aware_meaningful_v2(pred, record=record, request=request)
        reports.append(v2)
        comparisons.append((v1, v2.verdict))
        reasons.update(v2.reason_codes)
        explicit_labels = _available_labels(detail)
        for key, value in explicit_labels.items():
            labels[key].append((v1, v2.verdict, value))
        cases.append(
            {
                "id": case_id,
                "suite": suite,
                "prediction_status": "COMPLETE",
                "meaningful_program_v1": {
                    "status": "PASS" if v1 else "FAIL",
                    "verdict": v1,
                    "reason_code": reason,
                },
                "binding_aware_meaningful_v2": {
                    "status": (
                        "PASS"
                        if v2.verdict
                        else "FAIL" if v2.coverage_known else "UNKNOWN"
                    ),
                    **v2.to_dict(),
                },
                "effective_request": request.to_dict(),
                "labels": explicit_labels,
            }
        )
    summary = aggregate_meaning_reports_v2(reports)
    covered = sum(report.coverage_known for report in reports)
    positives = sum(report.verdict for report in reports)
    total = len(details)
    summary.update(
        {
            "n": total,
            "replayable_n": len(reports),
            "covered_n": covered,
            "strict_rate": positives / total if total else 0.0,
            "coverage_conditioned_rate": positives / covered if covered else 0.0,
            "coverage": covered / total if total else 0.0,
            "reason_prevalence": dict(sorted(reasons.items())),
        }
    )
    return {
        "label": label,
        "source": "committed_replay_bundle",
        "checkpoint_sha256s": [checkpoint_sha256] if not failures else [],
        "checkpoint_verifications": [verification],
        "replayable": bool(reports) and not failures,
        "failures": failures,
        "suite_counts": {suite: len(reports)},
        "v1_v2": _matrix(comparisons),
        "v2": summary,
        "reason_prevalence": dict(sorted(reasons.items())),
        "external_labels": {
            key: {
                "status": "available" if value else "UNKNOWN",
                "n": len(value),
                "v1_vs_label": _labeled_metrics(
                    [(v1, gold) for v1, _v2, gold in value]
                ),
                "v2_vs_label": _labeled_metrics(
                    [(v2, gold) for _v1, v2, gold in value]
                ),
            }
            for key, value in labels.items()
        },
        "cases": cases,
    }


def _parse_set(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    return label, Path(raw_path)


def _parse_note(value: str) -> tuple[str, str]:
    label, separator, note = value.partition("=")
    if not separator or not label or not note:
        raise argparse.ArgumentTypeError("expected LABEL=NOTE")
    return label, note


def _portable_output_paths(value: Any) -> Any:
    """Keep durable audit JSON checkout-independent while preserving run URIs."""
    if isinstance(value, dict):
        return {key: _portable_output_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_portable_output_paths(item) for item in value]
    if isinstance(value, str) and "/outputs/" in value:
        return "outputs/" + value.split("/outputs/", 1)[1]
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-set", action="append", type=_parse_set, default=[])
    parser.add_argument("--replay-bundle", type=Path)
    parser.add_argument("--capture-replay-bundle", type=Path)
    parser.add_argument("--unavailable-set", action="append", type=_parse_note, default=[])
    parser.add_argument("--gaming-corpus", type=Path)
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_EVAL_DATA_DIR)
    parser.add_argument("--minimum-frontier-sets", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args(argv)

    if args.replay_bundle:
        bundle = json.loads(args.replay_bundle.read_text(encoding="utf-8"))
        records = {
            key: ExampleRecord.from_dict(value)
            for key, value in (bundle.get("records") or {}).items()
        }
        sets = [_replay_bundle_set(row, records) for row in bundle.get("sets") or ()]
    else:
        sets = [
            _frontier_set(label, path, args.test_dir)
            for label, path in args.generation_set
        ]
    bundle_evidence = (
        {
            "path": str(args.replay_bundle),
            "sha256": hashlib.sha256(args.replay_bundle.read_bytes()).hexdigest(),
        }
        if args.replay_bundle
        else None
    )
    if args.capture_replay_bundle:
        if not args.generation_set:
            parser.error("--capture-replay-bundle requires --generation-set")
        bundle = _capture_replay_bundle(args.generation_set, args.test_dir)
        args.capture_replay_bundle.parent.mkdir(parents=True, exist_ok=True)
        args.capture_replay_bundle.write_text(
            json.dumps(_portable_output_paths(bundle), indent=2) + "\n",
            encoding="utf-8",
        )
        bundle_evidence = {
            "path": str(args.capture_replay_bundle),
            "sha256": hashlib.sha256(args.capture_replay_bundle.read_bytes()).hexdigest(),
        }
    if args.gaming_corpus:
        sets.append(_gaming_set(args.gaming_corpus))
    frontier = [row for row in sets if row["label"] != "deterministic_gaming_corpus"]
    replayable_frontier = [row for row in frontier if row["replayable"]]
    checkpoint_sha256s = {
        value
        for row in replayable_frontier
        for value in row.get("checkpoint_sha256s") or ()
    }
    blockers = []
    non_replayable = [row["label"] for row in sets if not row["replayable"]]
    if non_replayable:
        blockers.append(f"non_replayable_sets={','.join(non_replayable)}")
    if len(replayable_frontier) < args.minimum_frontier_sets:
        blockers.append(
            f"replayable_frontier_sets={len(replayable_frontier)} need>={args.minimum_frontier_sets}"
        )
    if len(checkpoint_sha256s) < args.minimum_frontier_sets:
        blockers.append(
            f"unique_checkpoint_sha256s={len(checkpoint_sha256s)} "
            f"need>={args.minimum_frontier_sets}"
        )
    if not sets:
        blockers.append("no_generation_or_gaming_sets_supplied")
    frontier_cases = [
        case
        for row in frontier
        for case in row.get("cases") or ()
        if case.get("prediction_status") == "COMPLETE"
    ]
    frontier_pairs = [
        (
            bool(case["meaningful_program_v1"]["verdict"]),
            bool(case["binding_aware_meaningful_v2"]["verdict"]),
        )
        for case in frontier_cases
    ]
    frontier_label_pairs: dict[str, list[tuple[bool, bool]]] = {}
    for case in frontier_cases:
        for name, gold in (case.get("labels") or {}).items():
            if isinstance(gold, bool):
                frontier_label_pairs.setdefault(name, []).append(
                    (bool(case["binding_aware_meaningful_v2"]["verdict"]), gold)
                )
    frontier_reasons = Counter(
        reason
        for row in frontier
        for reason, count in (row.get("reason_prevalence") or {}).items()
        for _ in range(int(count))
    )
    cases = [
        {
            "id": row["label"],
            "criteria": "Generation envelope is complete and v2 replay is deterministic.",
            "pass": row["replayable"],
            "failures": row.get("failures") or [],
            "result": row,
            "metadata": {"metric": "binding_aware_meaningful_v2"},
        }
        for row in sets
    ]
    if args.run_dir is None:
        parser.error("frontier/gaming audit mode requires --run-dir for AgentV")
    agentv = _portable_output_paths(
        publish_agentv_evaluation(
            args.run_dir,
            name="binding-aware-meaningful-v2-audit",
            claim="binding_aware_meaningful_v2_replay",
            cases=cases,
        )
    )
    payload = {
        "metric_family": {
            "historical": "meaningful_program_v1",
            "candidate": "binding_aware_meaningful_v2",
            "active_primary": "meaningful_program_v1",
            "v2_threshold": None,
        },
        "status": "complete" if not blockers else "blocked",
        "blockers": blockers,
        "minimum_frontier_sets": args.minimum_frontier_sets,
        "replayable_frontier_sets": len(replayable_frontier),
        "unique_checkpoint_sha256s": sorted(checkpoint_sha256s),
        "frontier_comparison": {
            "v1_vs_v2": _confusion(frontier_pairs),
            "reason_prevalence": dict(sorted(frontier_reasons.items())),
            "label_comparisons": {
                name: _labeled_metrics(frontier_label_pairs.get(name, []))
                for name in ("independent_judge", "human", "agentv", "efs0_04")
            },
        },
        "unavailable_named_sets": [
            {"label": label, "status": "UNKNOWN", "reason": reason}
            for label, reason in args.unavailable_set
        ],
        "sets": sets,
        "durable_replay_bundle": bundle_evidence
        or (
            {
                "path": str(args.replay_bundle),
                "sha256": hashlib.sha256(args.replay_bundle.read_bytes()).hexdigest(),
            }
            if args.replay_bundle
            else None
        ),
        "frontier_aggregate": {
            "v1_v2": _confusion(
                [
                    (
                        bool(case["meaningful_program_v1"]["verdict"]),
                        bool(case["binding_aware_meaningful_v2"]["verdict"]),
                    )
                    for row in replayable_frontier
                    for case in row["cases"]
                    if case["meaningful_program_v1"]["verdict"] is not None
                ]
            ),
            "external_labels": {
                label: {
                    "v1_vs_label": _labeled_metrics(
                        [
                            (bool(case["meaningful_program_v1"]["verdict"]), bool(case["labels"][label]))
                            for row in replayable_frontier
                            for case in row["cases"]
                            if label in case["labels"]
                        ]
                    ),
                    "v2_vs_label": _labeled_metrics(
                        [
                            (bool(case["binding_aware_meaningful_v2"]["verdict"]), bool(case["labels"][label]))
                            for row in replayable_frontier
                            for case in row["cases"]
                            if label in case["labels"]
                        ]
                    ),
                }
                for label in ("independent_judge", "human", "agentv", "efs0_04")
            },
        },
        "agentv": agentv,
        "caveats": [
            "UNKNOWN is never positive.",
            "EFS0-04 labels are unavailable until follow-on SLM-106 and remain UNKNOWN.",
            "No v1 threshold is copied to v2 before calibration.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "output": str(args.output)}))
    return 0 if not blockers else 2


if __name__ == "__main__":
    raise SystemExit(main())
