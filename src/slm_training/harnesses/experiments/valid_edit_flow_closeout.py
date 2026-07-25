"""Deterministic, fail-closed SLM-207 valid-edit-flow disposition."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slm_training.versioning import build_version_stamp

__all__ = [
    "CloseoutReport",
    "render_adr",
    "render_architecture",
    "render_markdown",
    "run_closeout",
]


@dataclass(frozen=True)
class EvidenceInput:
    issue: str
    path: str
    tier: str
    status: str
    consequence: str

    def to_dict(self, root: Path) -> dict[str, str]:
        target = root / self.path
        digest = hashlib.sha256(target.read_bytes()).hexdigest() if target.is_file() else "missing"
        return {
            "issue": self.issue,
            "path": self.path,
            "sha256": digest,
            "tier": self.tier,
            "status": self.status,
            "consequence": self.consequence,
        }


_INPUTS = (
    EvidenceInput("SLM-183", "docs/design/iter-slm183-power-protocol-20260720.md", "fixture", "blocked", "no powered confirmation authority"),
    EvidenceInput("SLM-185", "docs/design/iter-slm185-judge-resolution-20260720.md", "fixture", "blocked", "no production judge-resolution floor"),
    EvidenceInput("SLM-186", "docs/design/iter-slm186-verified-utility-20260721.md", "fixture", "blocked", "no deployment utility/canary authority"),
    EvidenceInput("SLM-192", "docs/design/iter-slm192-profile-flow-pipeline-20260721.md", "fixture", "blocked", "CPU fixture cost only"),
    EvidenceInput("SLM-193", "docs/design/iter-slm193-flow-caches-20260721.md", "fixture", "retain_as_control", "bit-exact cache wiring only"),
    EvidenceInput("SLM-194", "docs/design/iter-slm194-candidate-proposals-20260724.md", "fixture", "reject", "retain exact cached enumeration"),
    EvidenceInput("SLM-196", "docs/design/iter-slm196-legal-edit-bridge-20260723.md", "fixture", "blocked", "non-publishable two-target bridge corpus"),
    EvidenceInput("SLM-197", "docs/design/iter-slm197-direct-bridge-policy-20260723.md", "fixture", "inconclusive", "no frozen direct checkpoint"),
    EvidenceInput("SLM-200", "docs/design/iter-slm200-flow-objective-attribution-20260723.md", "fixture", "inconclusive", "no objective attribution conclusion"),
)


_LEVERS = (
    ("choice_ast", "choice/AST representation versus surface/production", "SLM-193", "retain_as_control", "exact representation substrate remains a control"),
    ("complete_bridges", "complete valid-state bridges versus local corruption", "SLM-196", "blocked", "only a non-publishable fixture corpus exists"),
    ("direct_policy", "direct legal-edit policy versus X22", "SLM-197", "inconclusive", "no frozen selected checkpoint"),
    ("time_conditioning", "time conditioning", "SLM-200", "inconclusive", "no powered matched attribution"),
    ("state_weighting", "state/loss weighting", "SLM-200", "inconclusive", "no powered matched attribution"),
    ("hazard_termination", "hazard/termination head", "SLM-200", "inconclusive", "no powered matched attribution"),
    ("flow_matching", "full edge-rate/flow matching", "SLM-200", "inconclusive", "SLM-200 did not license causal attribution"),
    ("few_step", "fixed-K/few-step distillation", "SLM-192", "blocked", "no selected few-step manifest"),
    ("batch_commits", "conflict-free batch commits", "SLM-192", "blocked", "no shared batch manifest"),
    ("cache", "exact state/candidate caches", "SLM-193", "retain_as_control", "bit-exact fixture contract only"),
    ("proposal", "learned candidate proposal", "SLM-194", "reject", "exact fallback avoided no final work"),
    ("bridge_curriculum", "bridge curriculum", "SLM-196", "blocked", "no publishable bridge corpus"),
    ("dagger", "one-round DAgger/on-policy correction", "SLM-196", "blocked", "no real selected model or powered corpus"),
    ("solver_trace", "solver-trace distillation", "SLM-196", "blocked", "no real selected model or powered corpus"),
    ("solver_only", "solver-only search", "SLM-192", "blocked", "no deployment solver ceiling"),
    ("hybrid", "learned-first/exact-late hybrid", "SLM-192", "blocked", "no learned candidate or solver ceiling"),
    ("utility", "semantic terminal utility, energy, and abstention", "SLM-186", "blocked", "utility and judge inputs remain fixture-only"),
    ("semantic_plan", "SemanticPlanV1 source/conditioning factors", "SLM-183", "blocked", "no oracle-gated confirmation authority"),
)


@dataclass(frozen=True)
class CloseoutReport:
    artifact_lock: list[dict[str, str]]
    dispositions: list[dict[str, str]]
    selected_stack: dict[str, str]
    version_stamp: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "ValidEditFlowCloseoutV1",
            "issue": "SLM-207",
            "status": "closeout",
            "claim_class": "blocked_or_fixture_evidence",
            "decision": "no_learned_flow_value_supported",
            "artifact_lock": self.artifact_lock,
            "dispositions": self.dispositions,
            "selected_stack": self.selected_stack,
            "unresolved_evidence_debt": [
                "publishable bridge corpus and replay coverage",
                "frozen selected checkpoint(s) with real-model cost and utility evidence",
                "powered confirmation targets with judge-resolution and single-touch authority",
            ],
            "version_stamp": self.version_stamp,
        }


def run_closeout(root: Path | str = ".") -> CloseoutReport:
    root = Path(root)
    locked = [item.to_dict(root) for item in _INPUTS]
    missing = [item["issue"] for item in locked if item["sha256"] == "missing"]
    if missing:
        raise FileNotFoundError(f"closeout evidence missing: {', '.join(missing)}")
    inputs_by_issue = {item["issue"]: item for item in locked}
    dispositions = []
    for lever_id, claim, issue, classification, reason in _LEVERS:
        evidence = inputs_by_issue[issue]
        dispositions.append(
            {
                "lever_id": lever_id,
                "claim": claim,
                "strongest_matched_control": "not evaluable from fixture-grade evidence",
                "primary_evidence_issue": issue,
                "primary_evidence_digest": evidence["sha256"],
                "evidence_tier": evidence["tier"],
                "confirmation_authority": "unavailable",
                "independent_targets_seeds_mde_precision": "unavailable",
                "primary_effect_ci_equivalence": "unavailable",
                "semantic_resolution_floor": "unavailable",
                "quality_and_goodhart": "blocked by SLM-185/SLM-186 fixture-only authority",
                "system_cost_effects": "unavailable outside CPU fixture evidence",
                "fidelity_and_safety": "exact compiler-owned candidate membership retained; no learned authority",
                "known_confounders": "fixture-only evidence and absent frozen checkpoint",
                "classification": classification,
                "reason": reason,
                "selected_action": "retain_control" if classification == "retain_as_control" else "default_off",
                "follow_up": "terminal until stated reopen conditions are met",
            }
        )
    return CloseoutReport(
        artifact_lock=locked,
        dispositions=dispositions,
        selected_stack={
            "learned_objective": "none",
            "runtime": "exact_cached_decoder_control",
            "default_action": "retain_default_off_research_paths",
            "guarantee": "No learned candidate is promoted; exact candidate membership remains compiler-owned.",
            "model": "none",
            "corpus": "none; no publishable bridge corpus",
            "source_conditioning": "none",
            "termination": "exact decoder control only",
            "cache": "SLM-193 bit-exact control",
            "proposal": "none; exact enumeration retained",
            "batch": "none",
            "solver": "none selected",
            "utility": "none; fixture-only authority",
            "checkpoint": "none",
        },
        version_stamp=build_version_stamp(
            "harness.experiments",
            "harness.experiments.valid_edit_flow_closeout",
            "matrix.valid_edit_flow_closeout",
        ),
    )


def render_markdown(report: CloseoutReport) -> str:
    lines = [
        "# Valid-edit flow closeout (SLM-207)",
        "",
        "**Decision:** `no_learned_flow_value_supported`. No learned objective or hybrid is selected.",
        "",
        "## Executive disposition",
        "",
        "The committed evidence is fixture-grade or explicitly blocked. It does not prove a deployment-level flow, direct-policy, DAgger, solver-trace, or hybrid advantage. Missing evidence is not treated as a negative result; affected levers remain blocked or inconclusive.",
        "",
        "Selected runtime: `exact_cached_decoder_control`; learned research paths remain default-off. No checkpoint, ship claim, or production default is authorized.",
        "",
        "## Causal table",
        "",
        "| Lever | Artifact | Tier | Classification | Default action | Reason |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in report.dispositions:
        lines.append(f"| {row['claim']} | {row['primary_evidence_issue']} | `{row['evidence_tier']}` | `{row['classification']}` | `{row['selected_action']}` | {row['reason']} |")
    lines.extend(["", "## Artifact lock", "", "| Issue | Tier | Status | Digest |", "| --- | --- | --- | --- |"])
    for item in report.artifact_lock:
        lines.append(f"| {item['issue']} | {item['tier']} | `{item['status']}` | `{item['sha256']}` |")
    lines.extend(["", "## Explicit non-guarantees", "", "There are no independent confirmation targets, seed counts, confidence intervals, equivalence results, cold/warm/tail deployment costs, optimality guarantees, or selected learned checkpoint in this closeout.", "", "## Reopen conditions", "", "A future closeout requires a publishable bridge corpus, frozen selected checkpoint(s), real-model cost and utility evidence, powered development targets, and untouched confirmation evidence. Until then, no branch may be promoted from this closeout."])
    return "\n".join(lines) + "\n"


def render_adr(report: CloseoutReport) -> str:
    return "\n".join(
        [
            "# ADR: Valid-edit flow closeout (SLM-207)",
            "",
            "## Status",
            "",
            "Accepted as a research closeout; it authorizes no production model or ship claim.",
            "",
            "## Decision",
            "",
            f"Select `{report.selected_stack['runtime']}` and `{report.selected_stack['learned_objective']}` learned objective.",
            "All learned research paths stay default-off.",
            "",
            "## Consequences",
            "",
            "Exact compiler-owned candidate membership remains the hard authority. Fixture evidence remains wiring evidence only; it is neither confirmation nor a negative result. Reopen only under the closeout's documented evidence conditions.",
            "",
        ]
    )


def render_architecture(report: CloseoutReport) -> str:
    return "\n".join(
        [
            "# Valid-edit flow authority map (SLM-207)",
            "",
            "```mermaid",
            "flowchart LR",
            "  Compiler[Compiler legal-candidate authority] --> Exact[Exact cached decoder control]",
            "  Exact --> Output[Verified output]",
            "  Learned[Learned flow/direct/hybrid research] -. default-off .-> Exact",
            "  Fixture[Fixture-grade evidence] -. no promotion authority .-> Learned",
            "```",
            "",
            f"Selected runtime: `{report.selected_stack['runtime']}`. Learned authority: none.",
            "",
        ]
    )
