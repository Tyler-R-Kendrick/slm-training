"""Synthesis feedback: turn a build's quality evidence into typed findings.

Every build closes its own loop: the quality report + rejection ledger are
distilled into per-family / per-synthesizer yields, dominant rejection
reasons, rule-based recommendations, and autoresearch-shaped experiment
candidates targeting the synthesis harness. The artifact
(``synthesis_feedback.json``) is the input agents use to improve producers
and synthesizers — gates are never weakened to make the numbers pass.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from slm_training.dsl.schema import ExampleRecord

FEEDBACK_SCHEMA_VERSION = 1

# Recommendation thresholds (documented in the artifact for honesty).
_LOW_YIELD = 0.4
_HIGH_DUP_SHARE = 0.5
_MIN_GROUP_CANDIDATES = 8


def _family_of(entry: dict[str, Any]) -> str:
    detail = entry.get("detail") or {}
    if detail.get("source_family"):
        return str(detail["source_family"])
    record = entry.get("record") or {}
    meta = record.get("meta") or {}
    if meta.get("source_family"):
        return str(meta["source_family"])
    if record.get("source"):
        return str(record["source"])
    return "unknown"


def _synth_of(entry: dict[str, Any]) -> str | None:
    detail = entry.get("detail") or {}
    if detail.get("synth"):
        return str(detail["synth"])
    record = entry.get("record") or {}
    meta = record.get("meta") or {}
    return str(meta["synth"]) if meta.get("synth") else None


def _group_stats(
    admitted: list[ExampleRecord], rejections: list[dict[str, Any]]
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    families: dict[str, dict[str, Any]] = {}
    synths: dict[str, dict[str, Any]] = {}

    def bucket(store: dict[str, dict[str, Any]], key: str) -> dict[str, Any]:
        return store.setdefault(
            key,
            {"admitted": 0, "rejected": 0, "by_stage": {}, "by_reason": {}},
        )

    for record in admitted:
        meta = record.meta or {}
        family = str(meta.get("source_family") or record.source)
        bucket(families, family)["admitted"] += 1
        if meta.get("synth"):
            bucket(synths, str(meta["synth"]))["admitted"] += 1
    for entry in rejections:
        stage = str(entry.get("stage") or "unknown")
        reason = str(entry.get("reason") or stage)
        for store, key in (
            (families, _family_of(entry)),
            (synths, _synth_of(entry)),
        ):
            if key is None:
                continue
            group = bucket(store, key)
            group["rejected"] += 1
            group["by_stage"][stage] = group["by_stage"].get(stage, 0) + 1
            group["by_reason"][reason] = group["by_reason"].get(reason, 0) + 1

    for store in (families, synths):
        for group in store.values():
            candidates = group["admitted"] + group["rejected"]
            group["candidates"] = candidates
            group["yield"] = (
                round(group["admitted"] / candidates, 4) if candidates else None
            )
            group["by_stage"] = dict(sorted(group["by_stage"].items()))
            group["by_reason"] = dict(
                sorted(group["by_reason"].items(), key=lambda item: -item[1])[:8]
            )
    return families, synths


def _dup_share(group: dict[str, Any]) -> float:
    dup = group["by_stage"].get("dedup", 0) + group["by_stage"].get("exposure", 0)
    return dup / group["candidates"] if group["candidates"] else 0.0


def _recommendations(
    families: dict[str, dict[str, Any]],
    synths: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    recommendations: list[dict[str, Any]] = []
    for kind, store in (("family", families), ("synthesizer", synths)):
        for name, group in sorted(store.items()):
            # Eval leakage is flagged at ANY volume — one leak matters.
            decontam = group["by_stage"].get("decontamination", 0)
            if decontam:
                recommendations.append(
                    {
                        "code": "eval_leakage_source",
                        "target_kind": kind,
                        "target": name,
                        "evidence": {"decontamination_drops": decontam},
                        "suggestion": (
                            "this source emitted eval-overlapping or reserved-"
                            "structure candidates; audit its inputs for leakage"
                        ),
                    }
                )
            if group["candidates"] < _MIN_GROUP_CANDIDATES:
                continue
            dup_share = _dup_share(group)
            if dup_share >= _HIGH_DUP_SHARE:
                recommendations.append(
                    {
                        "code": "redundant_expansion",
                        "target_kind": kind,
                        "target": name,
                        "evidence": {
                            "dup_share": round(dup_share, 4),
                            "candidates": group["candidates"],
                        },
                        "suggestion": (
                            "most output is deduplicated away — reduce expansion "
                            "count or diversify templates/namespaces for this "
                            f"{kind}"
                        ),
                    }
                )
            elif group["yield"] is not None and group["yield"] < _LOW_YIELD:
                top_reason = next(iter(group["by_reason"]), None)
                recommendations.append(
                    {
                        "code": "low_yield",
                        "target_kind": kind,
                        "target": name,
                        "evidence": {
                            "yield": group["yield"],
                            "top_reason": top_reason,
                            "candidates": group["candidates"],
                        },
                        "suggestion": (
                            f"under {int(_LOW_YIELD * 100)}% of candidates were "
                            f"admitted (top reason: {top_reason}) — fix the "
                            f"producer/synthesizer instead of relaxing gates"
                        ),
                    }
                )
    return recommendations


def _experiment_candidates(
    recommendations: list[dict[str, Any]], version: str
) -> list[dict[str, Any]]:
    """Autoresearch-shaped hypotheses targeting the synthesis harness."""
    experiments: list[dict[str, Any]] = []
    for item in recommendations:
        target = f"{item['target_kind']}:{item['target']}"
        if item["code"] == "redundant_expansion":
            experiments.append(
                {
                    "hypothesis": (
                        f"Reducing expansion volume or widening template "
                        f"diversity for {target} raises admitted-per-candidate "
                        f"yield without lowering admitted count"
                    ),
                    "rationale": f"build {version}: {item['evidence']}",
                    "expected_effect": "dup_share falls below 0.5; admitted holds",
                    "falsification_criteria": (
                        "admitted count drops >10% or dup_share stays >=0.5"
                    ),
                    "knobs": ["max_children", "synthesizer", "namespace_augment"],
                }
            )
        elif item["code"] == "low_yield":
            experiments.append(
                {
                    "hypothesis": (
                        f"Fixing the dominant rejection reason for {target} "
                        f"raises its yield above {_LOW_YIELD}"
                    ),
                    "rationale": f"build {version}: {item['evidence']}",
                    "expected_effect": f"{target} yield >= {_LOW_YIELD}",
                    "falsification_criteria": "yield unchanged after producer fix",
                    "knobs": ["producer_inputs", "synthesizer"],
                }
            )
        elif item["code"] == "eval_leakage_source":
            experiments.append(
                {
                    "hypothesis": (
                        f"{target} inputs contain eval-adjacent material; "
                        f"filtering its source removes decontamination drops"
                    ),
                    "rationale": f"build {version}: {item['evidence']}",
                    "expected_effect": "decontamination drops reach 0 for this source",
                    "falsification_criteria": "drops persist after source filtering",
                    "knobs": ["producer_inputs", "decontam_eval_root"],
                }
            )
    return experiments


def build_synthesis_feedback(
    *,
    version: str,
    profile: str,
    built_at: str,
    admitted: list[ExampleRecord],
    rejections: list[dict[str, Any]],
    quality_report: dict[str, Any],
) -> dict[str, Any]:
    families, synths = _group_stats(admitted, rejections)
    recommendations = _recommendations(families, synths)
    return {
        "schema_version": FEEDBACK_SCHEMA_VERSION,
        "version": version,
        "profile": profile,
        "built_at": built_at,
        "thresholds": {
            "low_yield": _LOW_YIELD,
            "high_dup_share": _HIGH_DUP_SHARE,
            "min_group_candidates": _MIN_GROUP_CANDIDATES,
        },
        "families": dict(sorted(families.items())),
        "synthesizers": dict(sorted(synths.items())),
        "warnings": quality_report.get("warnings") or [],
        "recommendations": recommendations,
        "experiment_candidates": _experiment_candidates(recommendations, version),
    }


def write_synthesis_feedback(out_dir: Path, feedback: dict[str, Any]) -> Path:
    path = out_dir / "synthesis_feedback.json"
    path.write_text(json.dumps(feedback, indent=2) + "\n", encoding="utf-8")
    return path


__all__ = [
    "FEEDBACK_SCHEMA_VERSION",
    "build_synthesis_feedback",
    "write_synthesis_feedback",
]
