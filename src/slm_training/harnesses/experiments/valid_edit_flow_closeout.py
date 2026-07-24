"""Deterministic, fail-closed SLM-207 valid-edit-flow disposition."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slm_training.versioning import build_version_stamp

__all__ = ["CloseoutReport", "render_markdown", "run_closeout"]


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
    ("choice_ast", "choice/AST representation", "retain_as_control", "exact representation substrate remains a control"),
    ("complete_bridges", "complete valid-state bridges", "blocked", "only a non-publishable fixture corpus exists"),
    ("direct_policy", "direct legal-edit policy", "inconclusive", "no frozen selected checkpoint"),
    ("time_weight_hazard", "time, weighting, and termination", "inconclusive", "no powered matched attribution"),
    ("flow_matching", "full edge-rate/flow matching", "inconclusive", "SLM-200 did not license causal attribution"),
    ("few_step_batch", "few-step distillation and batch commits", "blocked", "no selected few-step or shared batch manifest"),
    ("cache", "exact state/candidate caches", "retain_as_control", "bit-exact fixture contract only"),
    ("proposal", "learned candidate proposal", "reject", "exact fallback avoided no final work"),
    ("dagger_trace", "on-policy and solver-trace distillation", "blocked", "no real selected model or powered corpus"),
    ("solver_hybrid", "solver-only and learned/exact hybrid", "blocked", "no deployment solver ceiling or learned candidate"),
    ("utility", "terminal utility, energy, and abstention", "blocked", "utility and judge inputs remain fixture-only"),
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
            "version_stamp": self.version_stamp,
        }


def run_closeout(root: Path | str = ".") -> CloseoutReport:
    root = Path(root)
    locked = [item.to_dict(root) for item in _INPUTS]
    missing = [item["issue"] for item in locked if item["sha256"] == "missing"]
    if missing:
        raise FileNotFoundError(f"closeout evidence missing: {', '.join(missing)}")
    dispositions = [
        {"lever_id": lever_id, "claim": claim, "classification": classification, "reason": reason}
        for lever_id, claim, classification, reason in _LEVERS
    ]
    return CloseoutReport(
        artifact_lock=locked,
        dispositions=dispositions,
        selected_stack={
            "learned_objective": "none",
            "runtime": "exact_cached_decoder_control",
            "default_action": "retain_default_off_research_paths",
            "guarantee": "No learned candidate is promoted; exact candidate membership remains compiler-owned.",
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
        "The committed evidence is fixture-grade or explicitly blocked. It proves no deployment-level flow, direct-policy, DAgger, solver-trace, or hybrid advantage. Missing evidence is not treated as a negative result; affected levers remain blocked or inconclusive.",
        "",
        "Selected runtime: `exact_cached_decoder_control`; learned research paths remain default-off. No checkpoint, ship claim, or production default is authorized.",
        "",
        "## Causal table",
        "",
        "| Lever | Classification | Reason |",
        "| --- | --- | --- |",
    ]
    for row in report.dispositions:
        lines.append(f"| {row['claim']} | `{row['classification']}` | {row['reason']} |")
    lines.extend(["", "## Artifact lock", "", "| Issue | Tier | Status | Digest |", "| --- | --- | --- | --- |"])
    for item in report.artifact_lock:
        lines.append(f"| {item['issue']} | {item['tier']} | `{item['status']}` | `{item['sha256']}` |")
    lines.extend(["", "## Reopen conditions", "", "A future closeout requires a publishable bridge corpus, frozen selected checkpoint(s), real-model cost and utility evidence, powered development targets, and untouched confirmation evidence. Until then, no branch may be promoted from this closeout."])
    return "\n".join(lines) + "\n"
