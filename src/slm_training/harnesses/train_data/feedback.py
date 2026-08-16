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

FEEDBACK_SCHEMA_VERSION = 2

FINDING_CODES = (
    "low_yield",
    "redundant_expansion",
    "eval_leakage_source",
    "insufficient_unique_roots",
    "insufficient_semantic_frames",
    "coverage_cell_gap",
    "ambiguity_family_gap",
    "template_concentration",
    "renderer_concentration",
    "technical_trigger_concentration",
    "target_surface_leakage",
    "complete_inventory_leakage",
    "cue_only_shortcut",
    "source_only_shortcut",
    "prompt_target_contradiction",
    "counterfactual_coverage_gap",
    "paraphrase_invariance_gap",
    "split_family_leakage",
    "synthetic_anchor_deficit",
    "rare_family_erasure",
    "hard_forbidden_cue",
    "prefix_concentration",
    "pair_quality_auditor_unavailable",
    "sample_size_below_coverage_floor",
    "sample_size_above_capacity_ceiling",
    "unknown_blocking_reason",
)

# Findings that cannot close with a prose note; they need a new build receipt.
BLOCKING_FINDING_CODES = frozenset(
    {
        "insufficient_unique_roots",
        "cue_only_shortcut",
        "source_only_shortcut",
        "target_surface_leakage",
        "complete_inventory_leakage",
        "coverage_cell_gap",
        "synthetic_anchor_deficit",
        "split_family_leakage",
        "eval_leakage_source",
        "pair_quality_auditor_unavailable",
        "unknown_blocking_reason",
    }
)

# audit_corpus blocking_reasons → synthesis-feedback finding codes.
AUDITOR_REASON_TO_FINDING = {
    "cue_only_lift": "cue_only_shortcut",
    "source_only_lift": "source_only_shortcut",
    "target_surface_leak": "target_surface_leakage",
    "complete_inventory_leak": "complete_inventory_leakage",
    "hard_forbidden_cue": "hard_forbidden_cue",
    "prefix_concentration": "prefix_concentration",
    "template_concentration": "template_concentration",
    "prompt_target_contradiction": "prompt_target_contradiction",
    "pair_quality_auditor_unavailable": "pair_quality_auditor_unavailable",
}

_EXECUTABLE_KNOBS = {
    "insufficient_unique_roots": ["data_generation.unique_root_target", "data_source"],
    "coverage_cell_gap": ["data_generation.generator_max_depth", "data_generation.generator_max_width"],
    "template_concentration": ["data_generation.prompts_per_root", "data_generation.renderer_families"],
    "cue_only_shortcut": ["data_generation.prompt_surface", "data_generation.renderer_families"],
    "source_only_shortcut": ["data_generation.retain_trusted_anchors"],
    "synthetic_anchor_deficit": ["data_generation.retain_trusted_anchors"],
    "target_surface_leakage": ["data_generation.prompt_surface"],
    "complete_inventory_leakage": ["data_generation.prompt_surface"],
    "hard_forbidden_cue": [
        "data_generation.prompt_surface",
        "data_generation.renderer_families",
    ],
    "prefix_concentration": [
        "data_generation.prompts_per_root",
        "data_generation.renderer_families",
    ],
    "low_yield": ["synthesizer"],
    "redundant_expansion": ["max_records_per_parent", "synthesizer"],
    "eval_leakage_source": ["decontam_eval_root"],
    # Sample-adequacy climb signals (autoresearch.sample_adequacy).
    # Under-witnessed components get a *targeted* fail-closed rebuild
    # (until_coverage + raised component minimum), never a blind global
    # volume raise; measured-flat marginal gain routes away from volume —
    # never an uncharged capacity bump (decode-invariants VI).
    "sample_size_below_coverage_floor": [
        "data_generation.component_coverage_minimum",
        "data_generation.generation_mode",
        "data_generation.unique_root_target",
    ],
    "sample_size_above_capacity_ceiling": [
        "max_records_per_parent",
        "synthesizer",
    ],
}

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
        if stage == "selection":
            continue
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
                    "knobs": ["max_children", "synthesizer"],
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
                    "knobs": ["decontam_eval_root"],
                }
            )
    return experiments


def _capability_findings(quality_report: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    pair = quality_report.get("pair_quality") or {}
    roots = quality_report.get("unique_roots") or {}
    requested = roots.get("requested")
    admitted = roots.get("admitted")
    if (
        isinstance(requested, int)
        and isinstance(admitted, int)
        and admitted < requested
    ):
        findings.append(
            {
                "code": "insufficient_unique_roots",
                "severity": "hard_admission_failure",
                "authority": "blocks_capability_claim",
                "closure": "new_immutable_build",
                "evidence": {"requested": requested, "admitted": admitted},
            }
        )
    if pair.get("passed") is False:
        for reason in pair.get("blocking_reasons") or ():
            mapped = AUDITOR_REASON_TO_FINDING.get(str(reason), str(reason))
            code = (
                mapped if mapped in FINDING_CODES else "unknown_blocking_reason"
            )
            findings.append(
                {
                    "code": code,
                    "severity": "hard_admission_failure"
                    if code in BLOCKING_FINDING_CODES
                    else "experiment_recommendation",
                    "authority": "blocks_capability_claim",
                    "closure": "new_immutable_build",
                    "evidence": {"reason": reason},
                }
            )
    anchors = quality_report.get("source_anchors") or {}
    if anchors.get("deficit"):
        findings.append(
            {
                "code": "synthetic_anchor_deficit",
                "severity": "hard_admission_failure",
                "authority": "blocks_capability_claim",
                "waiver": "diagnostic_waiver_not_promotion",
                "closure": "new_immutable_build",
                "evidence": dict(anchors),
            }
        )
    return findings


def _capability_experiments(
    findings: list[dict[str, Any]], version: str
) -> list[dict[str, Any]]:
    experiments: list[dict[str, Any]] = []
    for finding in findings:
        knobs = _EXECUTABLE_KNOBS.get(finding["code"], ["data_generation.unique_root_target"])
        experiments.append(
            {
                "hypothesis": (
                    f"A matched data-only rebuild addressing {finding['code']} "
                    f"raises the named metric without changing model knobs"
                ),
                "rationale": f"build {version}: {finding.get('evidence')}",
                "expected_effect": f"{finding['code']} closes on a new manifest",
                "falsification_criteria": (
                    "finding persists on the rebuilt snapshot at the same policy hash"
                ),
                "knobs": knobs,
                "data_only": True,
            }
        )
    return experiments


def action_receipt(
    *,
    prior_feedback_hash: str,
    action_code: str,
    synthesis_plan_hash: str,
    dataset_manifest_hash: str,
    before: dict[str, Any],
    after: dict[str, Any],
    disposition: str,
    version_stamp: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Append-only receipt. A prose note is not a valid substitute."""

    if action_code in BLOCKING_FINDING_CODES and disposition == "acknowledged":
        raise ValueError(
            f"{action_code} cannot close with a prose acknowledgement; "
            "emit a new immutable build or an explicit diagnostic waiver"
        )
    return {
        "schema_version": "synthesis_feedback_receipt/v1",
        "prior_feedback_hash": prior_feedback_hash,
        "action_code": action_code,
        "synthesis_plan_hash": synthesis_plan_hash,
        "dataset_manifest_hash": dataset_manifest_hash,
        "before": before,
        "after": after,
        "disposition": disposition,
        "version_stamp": version_stamp,
    }


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
    findings = _capability_findings(quality_report)
    experiments = _experiment_candidates(recommendations, version)
    experiments.extend(_capability_experiments(findings, version))
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
        "findings": findings,
        "experiment_candidates": experiments,
        "claim_class": str(quality_report.get("claim_class") or "fixture_wiring"),
        # This builder cannot issue a capability certificate.
        "capability_certificate": False,
    }


def write_synthesis_feedback(out_dir: Path, feedback: dict[str, Any]) -> Path:
    path = out_dir / "synthesis_feedback.json"
    path.write_text(json.dumps(feedback, indent=2) + "\n", encoding="utf-8")
    return path


__all__ = [
    "AUDITOR_REASON_TO_FINDING",
    "BLOCKING_FINDING_CODES",
    "FEEDBACK_SCHEMA_VERSION",
    "FINDING_CODES",
    "action_receipt",
    "build_synthesis_feedback",
    "write_synthesis_feedback",
]
