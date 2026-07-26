"""Claim-separated DSH3-33 disposition over rebased CAP2 evidence."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class RebasedCap2Claim(str, Enum):
    RUNTIME_CORRECTNESS = "runtime_correctness"
    MODEL_VIEW_HYGIENE = "model_view_hygiene"
    TRAINING_VALIDITY = "training_validity"
    COMPLETE_PARTIAL_SAFETY = "complete_partial_safety"
    STOP_CALIBRATION = "stop_calibration"
    PERMUTATION_INVARIANCE = "permutation_invariance"
    HELD_OUT_SEMANTICS = "held_out_semantics"
    SYSTEMS_EFFICIENCY = "systems_efficiency"
    CAP0_RETENTION = "cap0_retention"
    CAP1_RETENTION = "cap1_retention"


class RebasedCap2Verdict(str, Enum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    NEGATIVE = "negative"
    INCONCLUSIVE = "inconclusive"
    INVALID = "invalid"
    FIXTURE_ONLY = "fixture_only"
    UNRUN_CONDITIONAL = "unrun_conditional"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class RebasedCap2EvidenceV1:
    evidence_id: str
    artifact_path: str
    report_schema: str
    code_commit: str
    version_stamp: Mapping[str, Any]
    data_identity: Mapping[str, Any]
    suite_identity: Mapping[str, Any]
    checkpoint_identity: str | None
    tokenizer_identity: str | None
    registry_identity: str | None
    policy_view_schema: str | None
    objective_identity: str | None
    decode_policy: str
    hardware_identity: Mapping[str, Any]
    result_identity: Mapping[str, Any]
    schema: str = "rebased_cap2_evidence/v1"

    def __post_init__(self) -> None:
        if not all((self.evidence_id, self.artifact_path, self.report_schema, self.code_commit, self.decode_policy)):
            raise ValueError("rebased CAP2 evidence requires stable identity")
        if self.version_stamp.get("stamp_schema") != "version_stamp/v1":
            raise ValueError("rebased CAP2 evidence requires a version stamp")
        if not all((self.data_identity, self.suite_identity, self.hardware_identity, self.result_identity)):
            raise ValueError("rebased CAP2 evidence requires complete identity fields")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "evidence_id": self.evidence_id,
            "artifact_path": self.artifact_path,
            "report_schema": self.report_schema,
            "code_commit": self.code_commit,
            "version_stamp": dict(self.version_stamp),
            "data_identity": dict(self.data_identity),
            "suite_identity": dict(self.suite_identity),
            "checkpoint_identity": self.checkpoint_identity,
            "tokenizer_identity": self.tokenizer_identity,
            "registry_identity": self.registry_identity,
            "policy_view_schema": self.policy_view_schema,
            "objective_identity": self.objective_identity,
            "decode_policy": self.decode_policy,
            "hardware_identity": dict(self.hardware_identity),
            "result_identity": dict(self.result_identity),
        }


@dataclass(frozen=True)
class RebasedCap2ClaimV1:
    claim: RebasedCap2Claim
    verdict: RebasedCap2Verdict
    reason: str
    evidence_ids: tuple[str, ...] = ()
    schema: str = "rebased_cap2_claim/v1"

    def __post_init__(self) -> None:
        if not self.reason:
            raise ValueError("rebased CAP2 claim requires a reason")
        if self.verdict is RebasedCap2Verdict.SUPPORTED and not self.evidence_ids:
            raise ValueError("supported claim requires evidence")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "claim": self.claim.value,
            "verdict": self.verdict.value,
            "reason": self.reason,
            "evidence_ids": list(self.evidence_ids),
        }


@dataclass(frozen=True)
class RebasedCap2OperatorPolicyDispositionV1:
    evidence: tuple[RebasedCap2EvidenceV1, ...]
    claims: tuple[RebasedCap2ClaimV1, ...]
    dsh5_may_start: bool
    allowed_heads: tuple[str, ...]
    allowed_objectives: tuple[str, ...]
    allowed_actions: tuple[str, ...]
    dsh5_reason: str
    historical_disposition: str
    version_stamp: Mapping[str, Any]
    schema: str = "rebased_cap2_operator_policy_disposition/v1"

    def __post_init__(self) -> None:
        if {item.claim for item in self.claims} != set(RebasedCap2Claim):
            raise ValueError("rebased disposition must cover each claim exactly once")
        evidence_ids = {item.evidence_id for item in self.evidence}
        if len(evidence_ids) != len(self.evidence):
            raise ValueError("rebased evidence IDs must be unique")
        if any(evidence_id not in evidence_ids for item in self.claims for evidence_id in item.evidence_ids):
            raise ValueError("rebased claim references unknown evidence")
        semantics = next(item for item in self.claims if item.claim is RebasedCap2Claim.HELD_OUT_SEMANTICS)
        if self.dsh5_may_start and semantics.verdict is not RebasedCap2Verdict.SUPPORTED:
            raise ValueError("DSH5 cannot start without supported held-out semantics")
        if not self.dsh5_may_start and any((self.allowed_heads, self.allowed_objectives, self.allowed_actions)):
            raise ValueError("rejected DSH5 disposition cannot allow policy inventory")
        if self.version_stamp.get("stamp_schema") != "version_stamp/v1":
            raise ValueError("rebased disposition requires a version stamp")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "evidence": [item.to_dict() for item in self.evidence],
            "claims": [item.to_dict() for item in self.claims],
            "dsh5": {
                "may_start": self.dsh5_may_start,
                "allowed_heads": list(self.allowed_heads),
                "allowed_objectives": list(self.allowed_objectives),
                "allowed_actions": list(self.allowed_actions),
                "reason": self.dsh5_reason,
            },
            "historical_disposition": self.historical_disposition,
            "version_stamp": dict(self.version_stamp),
        }


def _evidence(
    evidence_id: str, path: str, report: Mapping[str, Any], *, result: Mapping[str, Any], suite: Mapping[str, Any], data: Mapping[str, Any], policy_view: str | None = None, objective: str | None = None
) -> RebasedCap2EvidenceV1:
    stamp = report["version_stamp"]
    run = report.get("run", {})
    return RebasedCap2EvidenceV1(
        evidence_id=evidence_id,
        artifact_path=path,
        report_schema=str(report["schema"]),
        code_commit=str(stamp["code_commit"]),
        version_stamp=stamp,
        data_identity=data,
        suite_identity=suite,
        checkpoint_identity=run.get("checkpoint"),
        tokenizer_identity=None,
        registry_identity=None,
        policy_view_schema=policy_view,
        objective_identity=objective,
        decode_policy="compiler-owned legal set with singleton bypass and PARTIAL defer",
        hardware_identity={"device": run.get("device", "unknown"), "backend": run.get("backend", run.get("kind", "unknown"))},
        result_identity=result,
    )


def build_rebased_cap2_operator_policy_disposition(
    *,
    cap2_fixture: Mapping[str, Any],
    typed_policy: Mapping[str, Any],
    coverage: Mapping[str, Any],
    negative_ablation: Mapping[str, Any],
    failure_prediction: Mapping[str, Any],
    systems: Mapping[str, Any],
    permutation: Mapping[str, Any],
    version_stamp: Mapping[str, Any],
) -> RebasedCap2OperatorPolicyDispositionV1:
    if cap2_fixture.get("schema") != "cap2_operator_fixture_report/v1":
        raise ValueError("expected current CAP2 fixture report")
    if any(report.get("schema") != "dsh3_28_typed_operator_policy_report/v1" for report in (typed_policy, coverage, negative_ablation)):
        raise ValueError("expected typed policy reports")
    if failure_prediction.get("schema") != "operator_failure_prediction_report/v1":
        raise ValueError("expected failure-prediction report")
    if systems.get("schema") != "dsh3_32_operator_systems_benchmark_preflight/v1":
        raise ValueError("expected operator systems preflight")
    if permutation.get("schema") != "dsh3_27_operator_permutation_fixture/v1":
        raise ValueError("expected permutation report")
    if typed_policy["decision"]["verdict"] != "reject" or coverage["decision"]["verdict"] != "reject":
        raise ValueError("typed policy and coverage reports must retain their stop rules")
    if negative_ablation["decision"]["verdict"] != "reject":
        raise ValueError("negative ablation must retain its rejection")
    if failure_prediction["decision"]["verdict"] != "unavailable":
        raise ValueError("failure prediction must retain unavailable status")
    if systems["benchmark"]["verdict"] != "unavailable":
        raise ValueError("systems evidence must not be promoted")

    fixture_suite = cap2_fixture.get("suite", {})
    evidence = (
        _evidence("SLM-403.cap2-v2", "docs/design/dsh3-28-typed-operator-policy-20260726-cap2-v2-fiveheads/report.json", typed_policy, result=typed_policy["decision"], suite={"suite": "cap2_operator_v2", "n": typed_policy["counts"]["held_out_rows"]}, data=typed_policy["held_out_evidence"]["corpus"], policy_view="operator_policy_input/v1", objective="coverage-aware DEFER"),
        _evidence("SLM-404.coverage", "docs/design/dsh3-29-operator-ecoc-coverage-20260726-local/report.json", coverage, result=coverage["decision"], suite={"suite": "controlled PARTIAL", "n": coverage["counts"]["held_out_rows"]}, data=coverage["held_out_evidence"]["corpus"], policy_view="operator_policy_input/v1", objective="coverage-aware DEFER"),
        _evidence("SLM-405.negative-ablation", "docs/design/dsh3-30-collapse-negative-ablation-20260726-local/report.json", negative_ablation, result=negative_ablation["decision"], suite={"suite": "replay-negative strata", "n": negative_ablation["counts"]["held_out_rows"]}, data=negative_ablation["held_out_evidence"]["corpus"], policy_view="operator_policy_input/v1", objective="coverage-aware DEFER"),
        _evidence("SLM-406.failure-prediction", "docs/design/dsh3-31-operator-failure-prediction-20260726-local/report.json", failure_prediction, result=failure_prediction["decision"], suite={"suite": "local preflight", "n": len(failure_prediction.get("feature_rows", []))}, data={"sources": failure_prediction["sources"]}),
        _evidence("SLM-407.systems", "docs/design/dsh3-32-operator-serving-work-20260726-local/report.json", systems, result={"verdict": systems["benchmark"]["verdict"], "missing_arms": systems["benchmark"]["missing_arms"]}, suite={"suite": "operator-serving preflight", "n": systems["run"]["suite_n"]}, data={"context": systems["benchmark"]["identity"]["context_fingerprint"]}),
        _evidence("SLM-402.permutation", "docs/design/dsh3-27-operator-permutation-consistency.json", permutation, result={"verdict": "fixture_wiring_stop_rule_negative"}, suite={"suite": "operator permutation fixture", "n": permutation.get("run", {}).get("suite_n", 0)}, data={"kind": "fixture"}),
        _evidence("SLM-381.cap2-v2-fixture", "docs/design/dsh3-28-cap2-operator-v2-20260726/report.json", cap2_fixture, result={"honesty": cap2_fixture["run"]["honesty"]}, suite={"suite_version": fixture_suite.get("suite_version", "cap2_operator_v2"), "suite_n": cap2_fixture["run"]["suite_n"]}, data={"matrix_set": cap2_fixture["run"]["matrix_set"]}),
    )
    claims = (
        RebasedCap2ClaimV1(RebasedCap2Claim.RUNTIME_CORRECTNESS, RebasedCap2Verdict.FIXTURE_ONLY, "Current CAP2 v2 replay is fixture wiring evidence, not learned capability.", ("SLM-381.cap2-v2-fixture",)),
        RebasedCap2ClaimV1(RebasedCap2Claim.MODEL_VIEW_HYGIENE, RebasedCap2Verdict.FIXTURE_ONLY, "The sanitized policy-view boundary is exercised only by bounded local evidence.", ("SLM-403.cap2-v2",)),
        RebasedCap2ClaimV1(RebasedCap2Claim.TRAINING_VALIDITY, RebasedCap2Verdict.NEGATIVE, typed_policy["decision"]["reason"], ("SLM-403.cap2-v2",)),
        RebasedCap2ClaimV1(RebasedCap2Claim.COMPLETE_PARTIAL_SAFETY, RebasedCap2Verdict.NEGATIVE, coverage["decision"]["reason"], ("SLM-404.coverage",)),
        RebasedCap2ClaimV1(RebasedCap2Claim.STOP_CALIBRATION, RebasedCap2Verdict.UNRUN_CONDITIONAL, "No current powered stop/action comparison is compatible with the rejected typed-policy run."),
        RebasedCap2ClaimV1(RebasedCap2Claim.PERMUTATION_INVARIANCE, RebasedCap2Verdict.FIXTURE_ONLY, "Permutation evidence is fixture wiring only and cannot establish held-out benefit.", ("SLM-402.permutation",)),
        RebasedCap2ClaimV1(RebasedCap2Claim.HELD_OUT_SEMANTICS, RebasedCap2Verdict.NEGATIVE, typed_policy["decision"]["reason"], ("SLM-403.cap2-v2", "SLM-405.negative-ablation")),
        RebasedCap2ClaimV1(RebasedCap2Claim.SYSTEMS_EFFICIENCY, RebasedCap2Verdict.UNAVAILABLE, "No compatible five-arm measured serving comparison exists.", ("SLM-407.systems",)),
        RebasedCap2ClaimV1(RebasedCap2Claim.CAP0_RETENTION, RebasedCap2Verdict.UNRUN_CONDITIONAL, "No learned CAP2 candidate reached a retention evaluation."),
        RebasedCap2ClaimV1(RebasedCap2Claim.CAP1_RETENTION, RebasedCap2Verdict.UNAVAILABLE, "No CAP1-compatible learned candidate or retention artifact exists."),
    )
    return RebasedCap2OperatorPolicyDispositionV1(
        evidence=evidence,
        claims=claims,
        dsh5_may_start=False,
        allowed_heads=(),
        allowed_objectives=(),
        allowed_actions=(),
        dsh5_reason="No typed action form improved held-out CAP2 semantics; DSH5 remains closed.",
        historical_disposition="docs/design/dsh3-17-cap2-disposition-20260723/report.json (historical, immutable)",
        version_stamp=version_stamp,
    )
