"""Dataset quality report + persisted rejection ledger for train-data builds.

Every build writes two auditable artifacts beside ``records.jsonl``:

- ``rejected.jsonl`` — one line per rejected candidate with the stage and
  reason (verifier-in-the-loop style: nothing is dropped silently). Stages
  whose content is worth mining later (parse/contract failures, quarantines,
  quality fails) carry the full record payload; dedup/exposure drops are
  id-only because a surviving near-twin keeps the content in records.jsonl.
- ``quality_report.json`` — constraint-fitness, garbage, redundancy, and
  decontamination metrics for the admitted corpus.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from slm_training.dsl.schema import ExampleRecord

REPORT_SCHEMA_VERSION = 1

# Stages that persist the full candidate payload into rejected.jsonl.
_PAYLOAD_STAGES = frozenset(
    {"normalize", "selection", "verification", "verification_tier", "quality"}
)

# assess_record reasons that indicate a placeholder/template-contract breach
# rather than generic low quality.
_PLACEHOLDER_REASONS = frozenset(
    {
        "too_few_placeholders",
        "non_placeholder_string",
    }
)
_JUDGE_REASON_PREFIXES = (
    "prompt_component_missing_from_output",
    "schema_required_value_missing",
    "schema_value_role_mismatch",
    "schema_parser_error",
    "identity_echo_mismatch",
    "semantic_contract_mismatch",
    "boolean_literal_missing",
)


def rejection_entry(
    stage: str,
    reason: str,
    *,
    record: ExampleRecord | None = None,
    record_id: str | None = None,
    detail: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize one rejection into the rejected.jsonl row shape."""
    entry: dict[str, Any] = {
        "id": record_id or (record.id if record is not None else None),
        "stage": stage,
        "reason": reason,
    }
    if detail:
        entry["detail"] = detail
    if record is not None and stage in _PAYLOAD_STAGES:
        entry["record"] = record.to_dict()
    return entry


def write_rejected(out_dir: Path, entries: list[dict[str, Any]]) -> Path:
    path = out_dir / "rejected.jsonl"
    ordered = sorted(entries, key=lambda e: (str(e.get("stage")), str(e.get("id"))))
    path.write_text(
        "".join(json.dumps(entry, sort_keys=True) + "\n" for entry in ordered),
        encoding="utf-8",
    )
    return path


def _score_histogram(scores: list[float]) -> dict[str, int]:
    buckets = {"<0.4": 0, "0.4-0.55": 0, "0.55-0.7": 0, "0.7-0.85": 0, ">=0.85": 0}
    for score in scores:
        if score < 0.4:
            buckets["<0.4"] += 1
        elif score < 0.55:
            buckets["0.4-0.55"] += 1
        elif score < 0.7:
            buckets["0.55-0.7"] += 1
        elif score < 0.85:
            buckets["0.7-0.85"] += 1
        else:
            buckets[">=0.85"] += 1
    return buckets


def _stage_histogram(rejections: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in rejections:
        stage = str(entry.get("stage"))
        counts[stage] = counts.get(stage, 0) + 1
    return dict(sorted(counts.items()))


def _reason_histogram(rejections: list[dict[str, Any]], stage: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in rejections:
        if entry.get("stage") != stage:
            continue
        detail = entry.get("detail") or {}
        reasons = detail.get("reasons") or [entry.get("reason")]
        for reason in reasons:
            key = str(reason)
            counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _tier_histogram(records: list[ExampleRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        tier = str((record.meta or {}).get("verification_tier") or "unstamped")
        counts[tier] = counts.get(tier, 0) + 1
    return dict(sorted(counts.items()))


def _runtime_verified_fraction(records: list[ExampleRecord]) -> float | None:
    evaluated = 0
    passed = 0
    for record in records:
        verification = (record.meta or {}).get("verification") or {}
        for gate in verification.get("gates") or []:
            if str(gate.get("name")) != "runtime":
                continue
            status = str(gate.get("status"))
            if status in {"pass", "fail"}:
                evaluated += 1
                if status == "pass":
                    passed += 1
    if not evaluated:
        return None
    return round(passed / evaluated, 4)


def _top_clusters(
    records: list[ExampleRecord], *, limit: int = 10, samples: int = 5
) -> list[dict[str, Any]]:
    from slm_training.data.dedup import semantic_cluster_key

    groups: dict[tuple[str, str, str], list[str]] = {}
    for record in records:
        groups.setdefault(semantic_cluster_key(record), []).append(record.id)
    ranked = sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))
    return [
        {
            "cluster": "|".join(key),
            "count": len(ids),
            "sample_ids": sorted(ids)[:samples],
        }
        for key, ids in ranked[:limit]
    ]


def build_quality_report(
    *,
    version: str,
    profile: str,
    built_at: str,
    seed_count: int,
    collected_count: int,
    admitted: list[ExampleRecord],
    rejections: list[dict[str, Any]],
    source_error_count: int,
    cluster_exposure: dict[str, Any],
    per_family: list[dict[str, Any]],
    engines: dict[str, Any],
    decontamination_extra: dict[str, Any] | None = None,
    sanitization: dict[str, Any] | None = None,
) -> dict[str, Any]:
    by_stage = _stage_histogram(rejections)
    parse_failures = by_stage.get("normalize", 0)
    quality_reasons = _reason_histogram(rejections, "quality")
    dedup_reasons = _reason_histogram(rejections, "dedup")
    admitted_scores = [
        float((r.meta or {}).get("quality", {}).get("score") or 0.0) for r in admitted
    ]
    judge_flags = [
        bool((r.meta or {}).get("independent_judge_passed"))
        for r in admitted
        if "independent_judge_passed" in (r.meta or {})
    ]
    placeholder_violations = sum(
        count
        for reason, count in quality_reasons.items()
        if reason in _PLACEHOLDER_REASONS
    )
    judge_violations = sum(
        count
        for reason, count in quality_reasons.items()
        if reason.startswith(_JUDGE_REASON_PREFIXES)
    )
    candidate_count = collected_count + parse_failures
    decontamination = {
        "structure_reserved_rejected": by_stage.get("decontamination", 0),
    }
    if decontamination_extra:
        decontamination.update(decontamination_extra)

    top_clusters = _top_clusters(admitted)
    warnings: list[dict[str, Any]] = []
    admission_rate = len(admitted) / candidate_count if candidate_count else None
    if admission_rate is not None and admission_rate < 0.5:
        warnings.append(
            {
                "code": "high_rejection_rate",
                "value": round(admission_rate, 4),
                "message": "less than half of the candidates survived the gates; "
                "inspect rejected.jsonl before trusting this corpus",
            }
        )
    if len(admitted) >= 50 and top_clusters:
        top_share = top_clusters[0]["count"] / len(admitted)
        if top_share > 0.10:
            warnings.append(
                {
                    "code": "cluster_concentration",
                    "value": round(top_share, 4),
                    "message": "one semantic cluster holds over 10% of the corpus; "
                    "diversity is at risk (see redundancy.top_clusters)",
                }
            )
    flagged = int(decontamination.get("ngram_flagged") or 0)
    if flagged:
        warnings.append(
            {
                "code": "eval_overlap_flagged",
                "value": flagged,
                "message": "candidates overlapped eval suites by n-gram and were "
                "rejected; upstream producers may be leaking eval material",
            }
        )
    if placeholder_violations:
        warnings.append(
            {
                "code": "placeholder_contract_violations",
                "value": placeholder_violations,
                "message": "records breached the placeholder/template contract at "
                "the quality gate",
            }
        )
    judge_rate = round(sum(judge_flags) / len(judge_flags), 4) if judge_flags else None
    if judge_rate is not None and judge_rate < 1.0:
        warnings.append(
            {
                "code": "judge_failures_admitted",
                "value": judge_rate,
                "message": "admitted records include independent-judge failures "
                "(possible under the permissive profile)",
            }
        )
    sanitize_fallbacks = int((sanitization or {}).get("fallbacks") or 0)
    if sanitize_fallbacks:
        warnings.append(
            {
                "code": "sanitize_fallbacks",
                "value": sanitize_fallbacks,
                "message": "sanitization fell back to unchanged targets for some "
                "records; inspect sanitization.fallback_reasons",
            }
        )
    if (sanitization or {}).get("mode") == "audit" and profile == "strict":
        warnings.append(
            {
                "code": "sanitize_audit_only",
                "value": 1,
                "message": "the strict profile is running sanitization in audit "
                "mode; stored targets are not yet canonical/templatized",
            }
        )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "version": version,
        "profile": profile,
        "built_at": built_at,
        "counts": {
            "seeds": seed_count,
            "candidates": candidate_count,
            "collected": collected_count,
            "admitted": len(admitted),
            "rejected_total": len(rejections),
            "source_errors": source_error_count,
            "by_stage": by_stage,
        },
        "constraint_fitness": {
            "parse_failures": parse_failures,
            "parse_rate": (
                round(collected_count / candidate_count, 4) if candidate_count else None
            ),
            "tier_histogram": _tier_histogram(admitted),
            "judge_pass_rate": judge_rate,
            "quarantined": by_stage.get("verification", 0),
            "placeholder_contract_violations": placeholder_violations,
            "judge_contract_violations": judge_violations,
            "runtime_verified_fraction": _runtime_verified_fraction(admitted),
        },
        "garbage": {
            "quality_rejected": by_stage.get("quality", 0),
            "reason_histogram": quality_reasons,
            "admitted_score_histogram": _score_histogram(admitted_scores),
            "mean_quality_score": (
                round(sum(admitted_scores) / len(admitted_scores), 4)
                if admitted_scores
                else None
            ),
            "min_quality_score": min(admitted_scores) if admitted_scores else None,
        },
        "redundancy": {
            "dropped": {
                "exact_pair": dedup_reasons.get("exact_pair_duplicate", 0),
                "fuzzy_minhash": dedup_reasons.get("fuzzy_minhash", 0),
                "semantic_cluster_cap": dedup_reasons.get("semantic_cluster_cap", 0),
                "semantic_cosine": dedup_reasons.get("semantic_cosine", 0),
                "cross_corpus": dedup_reasons.get("cross_corpus_duplicate", 0),
                "max_records_per_parent": by_stage.get("exposure", 0),
            },
            "cluster_exposure": cluster_exposure,
            "top_clusters": top_clusters,
        },
        "decontamination": decontamination,
        "sanitization": sanitization or {"mode": "off"},
        "per_family": per_family,
        "engines": engines,
        "warnings": warnings,
    }


def write_quality_report(out_dir: Path, report: dict[str, Any]) -> Path:
    path = out_dir / "quality_report.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return path


__all__ = [
    "REPORT_SCHEMA_VERSION",
    "build_quality_report",
    "rejection_entry",
    "write_quality_report",
    "write_rejected",
]
