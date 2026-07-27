"""Claim-separated DSH5-12 disposition over DSH5 advanced-operator evidence.

DSH5 ("Bulk Operators, Transactions & Control Plane", milestones M1-M4) is a
loose portfolio of independent advanced-operator abstractions: exact bulk
selectors/actions (M1), operator transactions and a bounded set-valued
transaction policy (M2), branch sequence merge and a conversation
control-plane action set (M3), and an adaptive router, replay-grounded event
memory, and parameterized templates (M4). This module is the capstone
disposition (SLM-420 / DSH5-12): it binds one independent verdict per
abstraction so that a proven primitive (e.g. exact bulk atomicity) can never
be read as evidence for an unrelated, unrun claim (e.g. adaptive routing).

Mirrors the shape of ``slm_training.evals.cap2_operator_policy_disposition``
(SLM-408 / DSH3-33), which this disposition explicitly inherits from: DSH5's
allowed learned-policy inventory (heads/objectives/actions) is exactly the
empty inventory SLM-408 published, never re-derived here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from slm_training.harness_core.versioning import build_version_stamp


class AdvancedOperatorClaim(str, Enum):
    """The eleven advanced-operator abstractions DSH5-12 must verdict."""

    SELECTOR_CORRECTNESS = "selector_correctness"
    BULK_ATOMICITY = "bulk_atomicity"
    CROSSOVER_WORK = "crossover_work"
    TRANSACTION_CONTRACTS_EXECUTION = "transaction_contracts_execution"
    SET_VALUED_SELECTION = "set_valued_selection"
    SEQUENCE_MERGE = "sequence_merge"
    CONTROL_PLANE_EXECUTION_LEARNING = "control_plane_execution_learning"
    ADAPTIVE_ROUTING = "adaptive_routing"
    EVENT_MEMORY = "event_memory"
    PARAMETERIZED_TEMPLATES = "parameterized_templates"
    SYSTEMS_EFFICIENCY = "systems_efficiency"


class AdvancedOperatorVerdict(str, Enum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    NEGATIVE = "negative"
    INCONCLUSIVE = "inconclusive"
    INVALID = "invalid"
    FIXTURE_ONLY = "fixture_only"
    UNRUN_CONDITIONAL = "unrun_conditional"
    UNAVAILABLE = "unavailable"


class AdvancedOperatorDimension(str, Enum):
    """Independent axes a verdict may speak to. Success on one axis is never
    read as satisfying another -- the issue's own adversarial control."""

    RUNTIME_CORRECTNESS = "runtime_correctness"
    LEARNED_SEMANTIC_BENEFIT = "learned_semantic_benefit"
    PARTIAL_COVERAGE_SAFETY = "partial_coverage_safety"
    SYSTEMS_EFFICIENCY = "systems_efficiency"


_ALL_DIMENSIONS = frozenset(AdvancedOperatorDimension)

ALLOWED_RECOMMENDATIONS = frozenset(
    {
        "retain_as_compiler_utility",
        "continue_research",
        "enable_default_off_experiment",
        "eligible_for_promotion_review",
        "reject",
    }
)


@dataclass(frozen=True)
class AdvancedOperatorEvidenceV1:
    """One exact, cited artifact backing one or more claims."""

    evidence_id: str
    issue_alias: str
    linear_id: str
    artifact_path: str
    code_commit: str
    version_stamp: Mapping[str, Any]
    data_identity: Mapping[str, Any]
    suite_identity: Mapping[str, Any]
    checkpoint_identity: str | None
    tokenizer_identity: str | None
    registry_identity: str | None
    schema_identity: str | None
    objective_identity: str | None
    decode_policy: str
    hardware_identity: Mapping[str, Any]
    result_identity: Mapping[str, Any]
    schema: str = "advanced_operator_evidence/v1"

    def __post_init__(self) -> None:
        if not all(
            (
                self.evidence_id,
                self.issue_alias,
                self.linear_id,
                self.artifact_path,
                self.code_commit,
                self.decode_policy,
            )
        ):
            raise ValueError("advanced-operator evidence requires stable identity")
        if self.version_stamp.get("stamp_schema") != "version_stamp/v1":
            raise ValueError("advanced-operator evidence requires a version stamp")
        if not all((self.data_identity, self.suite_identity, self.hardware_identity, self.result_identity)):
            raise ValueError("advanced-operator evidence requires complete identity fields")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "evidence_id": self.evidence_id,
            "issue_alias": self.issue_alias,
            "linear_id": self.linear_id,
            "artifact_path": self.artifact_path,
            "code_commit": self.code_commit,
            "version_stamp": dict(self.version_stamp),
            "data_identity": dict(self.data_identity),
            "suite_identity": dict(self.suite_identity),
            "checkpoint_identity": self.checkpoint_identity,
            "tokenizer_identity": self.tokenizer_identity,
            "registry_identity": self.registry_identity,
            "schema_identity": self.schema_identity,
            "objective_identity": self.objective_identity,
            "decode_policy": self.decode_policy,
            "hardware_identity": dict(self.hardware_identity),
            "result_identity": dict(self.result_identity),
        }


@dataclass(frozen=True)
class AdvancedOperatorClaimV1:
    """One claim's headline verdict plus its full per-dimension breakdown.

    ``dimension_verdicts`` must name all four :class:`AdvancedOperatorDimension`
    values explicitly -- a claim can never leave an axis unaddressed, and the
    headline ``verdict`` must equal one of the axis verdicts it already
    asserted (never an invented fifth state that blends axes together).
    """

    claim: AdvancedOperatorClaim
    verdict: AdvancedOperatorVerdict
    reason: str
    dimension_verdicts: Mapping[AdvancedOperatorDimension, AdvancedOperatorVerdict]
    dimension_reasons: Mapping[AdvancedOperatorDimension, str]
    evidence_ids: tuple[str, ...] = ()
    schema: str = "advanced_operator_claim/v1"

    def __post_init__(self) -> None:
        if not self.reason:
            raise ValueError("advanced-operator claim requires a reason")
        if set(self.dimension_verdicts) != _ALL_DIMENSIONS:
            raise ValueError(
                "a claim must state a verdict for every dimension "
                "(runtime correctness, learned semantic benefit, "
                "partial-coverage safety, systems efficiency)"
            )
        if set(self.dimension_reasons) != _ALL_DIMENSIONS:
            raise ValueError("a claim must state a reason for every dimension")
        if any(not reason for reason in self.dimension_reasons.values()):
            raise ValueError("every dimension reason must be non-empty")
        if self.verdict not in self.dimension_verdicts.values():
            raise ValueError(
                "the headline verdict must equal one of the claim's own "
                "per-dimension verdicts -- it may never blend axes"
            )
        any_supported = self.verdict is AdvancedOperatorVerdict.SUPPORTED or any(
            v is AdvancedOperatorVerdict.SUPPORTED for v in self.dimension_verdicts.values()
        )
        if any_supported and not self.evidence_ids:
            raise ValueError("a supported axis requires cited evidence")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "claim": self.claim.value,
            "verdict": self.verdict.value,
            "reason": self.reason,
            "dimension_verdicts": {dim.value: v.value for dim, v in self.dimension_verdicts.items()},
            "dimension_reasons": {dim.value: r for dim, r in self.dimension_reasons.items()},
            "evidence_ids": list(self.evidence_ids),
        }


@dataclass(frozen=True)
class InheritedPolicyInventoryV1:
    """The exact allowed primitive policy/head/objective/action inventory
    inherited from the required DSH3-33 disposition (SLM-408) -- never
    re-derived here."""

    source_disposition: str
    source_artifact_path: str
    dsh5_may_start: bool
    allowed_heads: tuple[str, ...]
    allowed_objectives: tuple[str, ...]
    allowed_actions: tuple[str, ...]
    reason: str
    schema: str = "advanced_operator_inherited_policy_inventory/v1"

    def __post_init__(self) -> None:
        if not all((self.source_disposition, self.source_artifact_path, self.reason)):
            raise ValueError("inherited policy inventory requires its source disposition and reason")
        if not self.dsh5_may_start and any((self.allowed_heads, self.allowed_objectives, self.allowed_actions)):
            raise ValueError(
                "an inherited disposition that did not authorize DSH5 cannot "
                "carry an allowed learned-policy inventory"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "source_disposition": self.source_disposition,
            "source_artifact_path": self.source_artifact_path,
            "dsh5_may_start": self.dsh5_may_start,
            "allowed_heads": list(self.allowed_heads),
            "allowed_objectives": list(self.allowed_objectives),
            "allowed_actions": list(self.allowed_actions),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class RetentionClaimV1:
    """CAP0/CAP1/CAP2 continuity: does this disposition change the standing
    capability-certificate posture, or only confirm it unchanged?"""

    capability: str
    verdict: AdvancedOperatorVerdict
    reason: str
    evidence_ids: tuple[str, ...] = ()
    schema: str = "advanced_operator_retention_claim/v1"

    def __post_init__(self) -> None:
        if self.capability not in {"CAP0", "CAP1", "CAP2"}:
            raise ValueError("retention claim must name CAP0, CAP1, or CAP2")
        if not self.reason:
            raise ValueError("retention claim requires a reason")
        if self.verdict is AdvancedOperatorVerdict.SUPPORTED and not self.evidence_ids:
            raise ValueError("a supported retention claim requires cited evidence")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "capability": self.capability,
            "verdict": self.verdict.value,
            "reason": self.reason,
            "evidence_ids": list(self.evidence_ids),
        }


@dataclass(frozen=True)
class AdvancedOperatorDispositionV1:
    """The DSH5-12 / SLM-420 capstone disposition."""

    evidence: tuple[AdvancedOperatorEvidenceV1, ...]
    claims: tuple[AdvancedOperatorClaimV1, ...]
    inherited_policy: InheritedPolicyInventoryV1
    retention: tuple[RetentionClaimV1, ...]
    advanced_path_enabled_by_default: bool
    recommendation: str
    recommendation_reason: str
    historical_disposition: str
    version_stamp: Mapping[str, Any]
    schema: str = "advanced_operator_disposition/v1"

    def __post_init__(self) -> None:
        claim_kinds = [item.claim for item in self.claims]
        if set(claim_kinds) != set(AdvancedOperatorClaim):
            raise ValueError("disposition must cover each of the eleven claims exactly once")
        if len(set(claim_kinds)) != len(claim_kinds):
            raise ValueError("disposition has a duplicate claim")
        evidence_ids = {item.evidence_id for item in self.evidence}
        if len(evidence_ids) != len(self.evidence):
            raise ValueError("advanced-operator evidence IDs must be unique")
        if any(evidence_id not in evidence_ids for item in self.claims for evidence_id in item.evidence_ids):
            raise ValueError("a claim references unknown evidence")
        if {item.capability for item in self.retention} != {"CAP0", "CAP1", "CAP2"}:
            raise ValueError("disposition must cover CAP0, CAP1, and CAP2 retention exactly once each")
        if len({item.capability for item in self.retention}) != len(self.retention):
            raise ValueError("disposition has a duplicate retention claim")
        if any(evidence_id not in evidence_ids for item in self.retention for evidence_id in item.evidence_ids):
            raise ValueError("a retention claim references unknown evidence")
        if self.advanced_path_enabled_by_default:
            raise ValueError("no advanced path may be enabled by default from this disposition alone")
        if self.recommendation not in ALLOWED_RECOMMENDATIONS:
            raise ValueError(f"unsupported recommendation: {self.recommendation!r}")
        if not self.recommendation_reason:
            raise ValueError("disposition requires a recommendation reason")
        if not self.historical_disposition:
            raise ValueError("disposition must reference the historical DSH3-33 disposition, not rewrite it")
        if self.version_stamp.get("stamp_schema") != "version_stamp/v1":
            raise ValueError("disposition requires a version stamp")

    def claim(self, kind: AdvancedOperatorClaim) -> AdvancedOperatorClaimV1:
        return next(item for item in self.claims if item.claim is kind)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "evidence": [item.to_dict() for item in self.evidence],
            "claims": [item.to_dict() for item in self.claims],
            "inherited_policy": self.inherited_policy.to_dict(),
            "retention": [item.to_dict() for item in self.retention],
            "advanced_path_enabled_by_default": self.advanced_path_enabled_by_default,
            "recommendation": self.recommendation,
            "recommendation_reason": self.recommendation_reason,
            "historical_disposition": self.historical_disposition,
            "version_stamp": dict(self.version_stamp),
        }


# Commit hashes are pinned identity for historical, already-merged DSH5
# evidence -- never regenerated, mirroring cap2_operator_policy_rebase.py's
# ``pinned_baseline`` precedent.
_COMMITS: dict[str, str] = {
    "SLM-409": "64d508d0b28d2a695f2b49cd3362865c671dfeb7",
    "SLM-410": "228d2c5d78ad8d8273c5015445aa14a4c1b0014d",
    "SLM-412": "3e6d1d3636966c17b60a5f4c1adffbfa2750df87",
    "SLM-413": "99e07f2efe75a692f55908771cbf1f2a4b67cf80",
    "SLM-414": "7e1fc586b53dfb3e2fc6386a2aa46c79d9baf771",
    "SLM-415": "2cd177c41bf76119b4b2a01d4047bf7b9f7d9a2c",
    "SLM-416": "b40b705b4fdb11713be233333e1a1cad3fd813c1",
    "SLM-418": "78d598be52018648c92d858cccdb1b4051a981ec",
    "SLM-419": "8d04210375647bb69674b85db582df45eb7082aa",
}

_DECODE_POLICY = "compiler-owned legal set with singleton bypass and PARTIAL defer"


def _fixture_evidence(
    *,
    evidence_id: str,
    issue_alias: str,
    linear_id: str,
    artifact_path: str,
    component_id: str,
    version_stamp: Mapping[str, Any],
    suite: str,
    result: str,
) -> AdvancedOperatorEvidenceV1:
    """One compiler-contract/unit-fixture evidence item (no train/eval run)."""
    return AdvancedOperatorEvidenceV1(
        evidence_id=evidence_id,
        issue_alias=issue_alias,
        linear_id=linear_id,
        artifact_path=artifact_path,
        code_commit=_COMMITS[linear_id],
        version_stamp=version_stamp,
        data_identity={"kind": "openui_pack_fixture", "backend": "real_dsl_pack", "component": component_id},
        suite_identity={"suite": suite, "kind": "unit_fixture"},
        checkpoint_identity=None,
        tokenizer_identity=None,
        registry_identity=None,
        schema_identity=None,
        objective_identity=None,
        decode_policy=_DECODE_POLICY,
        hardware_identity={"device": "cpu", "backend": "pytest"},
        result_identity={"verdict": result},
    )


def _report_evidence(
    *,
    evidence_id: str,
    issue_alias: str,
    linear_id: str,
    artifact_path: str,
    report: Mapping[str, Any],
    suite: str,
    result: Mapping[str, Any],
) -> AdvancedOperatorEvidenceV1:
    """One evidence item backed by a real generated ``report.json``."""
    stamp = report["version_stamp"]
    run = report.get("run", {})
    return AdvancedOperatorEvidenceV1(
        evidence_id=evidence_id,
        issue_alias=issue_alias,
        linear_id=linear_id,
        artifact_path=artifact_path,
        code_commit=str(stamp["code_commit"]),
        version_stamp=stamp,
        data_identity={"report_schema": report.get("schema")},
        suite_identity={"suite": suite, "n": run.get("suite_n", 0)},
        checkpoint_identity=run.get("checkpoint"),
        tokenizer_identity=None,
        registry_identity=None,
        schema_identity=str(report.get("schema")) if report.get("schema") else None,
        objective_identity=run.get("primary_metric"),
        decode_policy=_DECODE_POLICY,
        hardware_identity={"device": run.get("device", "cpu"), "backend": run.get("backend", "unknown")},
        result_identity=dict(result),
    )


# Issue -> the real registry component that issue's own change owns. SLM-419
# (DSH5-11) is a repository-wide audit that touched no watched component, so
# it gets a bare stamp (empty ``components``) rather than a fabricated owner.
# The single owner of this mapping: both the publisher script and its test
# import it from here rather than keeping their own copies, so a registry
# rename in one place can't leave the other silently stale.
COMPONENT_BY_ISSUE = {
    "SLM-409": "dsl.operators.selectors",
    "SLM-410": "dsl.operators.bulk",
    "SLM-412": "dsl.operators.transactions",
    "SLM-413": "dsl.operators.transaction_executor",
    "SLM-414": "harness.experiments.operator_transaction_policy",
    "SLM-415": "dsl.operators.sequence_merge",
    "SLM-416": "dsl.operators.control_actions",
    "SLM-418": "dsl.operators.replay_preference",
}


def component_version_stamps() -> dict[str, dict[str, Any]]:
    """Real ``build_version_stamp(...)`` envelopes for every non-fixture DSH5 issue."""
    stamps = {issue: build_version_stamp(component) for issue, component in COMPONENT_BY_ISSUE.items()}
    stamps["SLM-419"] = build_version_stamp()
    return stamps


def build_advanced_operator_disposition(
    *,
    dsh3_33_report: Mapping[str, Any],
    dsh5_03_report: Mapping[str, Any],
    dsh5_09_report: Mapping[str, Any],
    version_stamp: Mapping[str, Any],
    component_version_stamps: Mapping[str, Mapping[str, Any]],
) -> AdvancedOperatorDispositionV1:
    """Build the DSH5-12 capstone disposition from durable, already-committed
    evidence.

    ``component_version_stamps`` maps a short evidence key (``SLM-409`` ..
    ``SLM-419``) to a real ``build_version_stamp(...)`` envelope over the
    actual registry component the corresponding issue owns -- never a
    fabricated identity.
    """
    if dsh3_33_report.get("schema") != "dsh3_33_rebased_cap2_disposition_report/v1":
        raise ValueError("expected the DSH3-33 (SLM-408) rebased CAP2 disposition report")
    dsh5_block = dsh3_33_report["disposition"]["dsh5"]
    if dsh5_block["may_start"] is not False:
        raise ValueError("DSH3-33 unexpectedly authorized DSH5 -- inherited inventory must be re-audited")

    # DSH5-03's own report shape is read defensively: this module never
    # requires an exact schema string it did not itself mint, but it must at
    # least carry the version stamp `_report_evidence` binds identity to.
    if "version_stamp" not in dsh5_03_report:
        raise ValueError(
            "DSH5-03 report is missing the version stamp required for evidence identity"
        )

    evidence = (
        _fixture_evidence(
            evidence_id="SLM-409.dsh5-01",
            issue_alias="DSH5-01",
            linear_id="SLM-409",
            artifact_path="docs/design/dsh5-01-selector-refs.md",
            component_id="dsl.operators.selectors",
            version_stamp=component_version_stamps["SLM-409"],
            suite="tests/test_dsl/test_operator_selectors.py",
            result="exact finite selector membership, freshness, fanout bound, and permutation invariance proven",
        ),
        _fixture_evidence(
            evidence_id="SLM-410.dsh5-02",
            issue_alias="DSH5-02",
            linear_id="SLM-410",
            artifact_path="docs/design/dsh5-02-bulk-set-property.md",
            component_id="dsl.operators.bulk",
            version_stamp=component_version_stamps["SLM-410"],
            suite="tests/test_dsl/test_bulk_operators.py",
            result="atomic bulk set-property proven structurally and by rejection-atomicity/replay/lowering tests",
        ),
        _report_evidence(
            evidence_id="SLM-411.dsh5-03",
            issue_alias="DSH5-03",
            linear_id="SLM-411",
            artifact_path="docs/design/dsh5-03-bulk-operator-crossover-20260726-local/report.json",
            report=dsh5_03_report,
            suite="dsh5-03 bulk-operator-crossover preflight",
            result={"verdict": "unavailable", "reason": "no matched five-arm serving rows exist"},
        ),
        _fixture_evidence(
            evidence_id="SLM-412.dsh5-04",
            issue_alias="DSH5-04",
            linear_id="SLM-412",
            artifact_path="docs/design/dsh5-04-operator-transactions.md",
            component_id="dsl.operators.transactions",
            version_stamp=component_version_stamps["SLM-412"],
            suite="tests/test_dsl/test_operator_transactions.py",
            result="one-base multi-action dependency/conflict contracts proven; no execution path",
        ),
        _fixture_evidence(
            evidence_id="SLM-413.dsh5-05",
            issue_alias="DSH5-05",
            linear_id="SLM-413",
            artifact_path="docs/design/dsh5-05-transaction-executor.md",
            component_id="dsl.operators.transaction_executor",
            version_stamp=component_version_stamps["SLM-413"],
            suite="tests/test_dsl/test_operator_transaction_executor.py",
            result="atomic N-way composition/commit/replay proven for disjoint and declared-commuting subsets",
        ),
        _fixture_evidence(
            evidence_id="SLM-414.dsh5-06",
            issue_alias="DSH5-06",
            linear_id="SLM-414",
            artifact_path="docs/design/dsh5-06-operator-transaction-policy.md",
            component_id="harness.experiments.operator_transaction_policy",
            version_stamp=component_version_stamps["SLM-414"],
            suite="tests/test_harnesses/experiments/test_operator_transaction_policy.py",
            result="bounded-K decode safety invariant held under adversarial corrupted-belief control; N=2 fixture only",
        ),
        _fixture_evidence(
            evidence_id="SLM-415.dsh5-07",
            issue_alias="DSH5-07",
            linear_id="SLM-415",
            artifact_path="docs/design/dsh5-07-branch-sequence-merge.md",
            component_id="dsl.operators.sequence_merge",
            version_stamp=component_version_stamps["SLM-415"],
            suite="tests/test_dsl/test_operator_sequence_merge.py",
            result="N-step conservative merge proven for AST_EDIT-only chains, excluding TRANSACTION_COMMIT/mid-sequence FORK",
        ),
        _fixture_evidence(
            evidence_id="SLM-416.dsh5-08",
            issue_alias="DSH5-08",
            linear_id="SLM-416",
            artifact_path="docs/design/dsh5-08-conversation-control-actions.md",
            component_id="dsl.operators.control_actions",
            version_stamp=component_version_stamps["SLM-416"],
            suite="tests/test_dsl/test_conversation_control_actions.py",
            result="compiler-owned control legal set, dispatch, and deterministic baseline proven; no learned policy trained",
        ),
        _report_evidence(
            evidence_id="SLM-417.dsh5-09",
            issue_alias="DSH5-09",
            linear_id="SLM-417",
            artifact_path="docs/design/dsh5-09-adaptive-router-20260726-local/report.json",
            report=dsh5_09_report,
            suite="dsh5-09 adaptive-router admission preflight",
            result=dict(dsh5_09_report.get("disposition", {})),
        ),
        _fixture_evidence(
            evidence_id="SLM-418.dsh5-10",
            issue_alias="DSH5-10",
            linear_id="SLM-418",
            artifact_path="docs/design/dsh5-10-replay-preference-rows.md",
            component_id="dsl.operators.replay_preference",
            version_stamp=component_version_stamps["SLM-418"],
            suite="tests/test_dsl/test_replay_preference.py",
            result="7 of 7 named patterns (edit-then-undo, undo-then-redo, partial-rollback, checkout-another-state, fork-then-choose-one-branch, merge-success, pronoun-focus-followup) extract and replay-verify; partial slice",
        ),
        _fixture_evidence(
            evidence_id="SLM-419.dsh5-11",
            issue_alias="DSH5-11",
            linear_id="SLM-419",
            artifact_path="docs/design/dsh5-11-parameterized-template-disposition.md",
            component_id="data.macro_induction",
            version_stamp=component_version_stamps["SLM-419"],
            suite="repository-wide tracked-path audit (TSA-013/TSA-014/template_manifest)",
            result="no TSA/parameterized-template artifact exists anywhere in the tracked repository",
        ),
        _report_evidence(
            evidence_id="SLM-408.dsh3-33",
            issue_alias="DSH3-33",
            linear_id="SLM-408",
            artifact_path="docs/design/dsh3-33-rebased-cap2-disposition-20260726-local/report.json",
            report=dsh3_33_report,
            suite="dsh3-33 rebased CAP2 operator-policy disposition",
            result={"dsh5": dsh5_block},
        ),
    )

    def dims(
        runtime: tuple[AdvancedOperatorVerdict, str],
        learned: tuple[AdvancedOperatorVerdict, str],
        partial: tuple[AdvancedOperatorVerdict, str],
        systems: tuple[AdvancedOperatorVerdict, str],
    ) -> tuple[Mapping[AdvancedOperatorDimension, AdvancedOperatorVerdict], Mapping[AdvancedOperatorDimension, str]]:
        verdicts = {
            AdvancedOperatorDimension.RUNTIME_CORRECTNESS: runtime[0],
            AdvancedOperatorDimension.LEARNED_SEMANTIC_BENEFIT: learned[0],
            AdvancedOperatorDimension.PARTIAL_COVERAGE_SAFETY: partial[0],
            AdvancedOperatorDimension.SYSTEMS_EFFICIENCY: systems[0],
        }
        reasons = {
            AdvancedOperatorDimension.RUNTIME_CORRECTNESS: runtime[1],
            AdvancedOperatorDimension.LEARNED_SEMANTIC_BENEFIT: learned[1],
            AdvancedOperatorDimension.PARTIAL_COVERAGE_SAFETY: partial[1],
            AdvancedOperatorDimension.SYSTEMS_EFFICIENCY: systems[1],
        }
        return verdicts, reasons

    S = AdvancedOperatorVerdict.SUPPORTED
    U = AdvancedOperatorVerdict.UNAVAILABLE
    R = AdvancedOperatorVerdict.UNRUN_CONDITIONAL
    F = AdvancedOperatorVerdict.FIXTURE_ONLY

    claims: list[AdvancedOperatorClaimV1] = []

    dv, dr = dims(
        (S, "Exact finite membership, freshness, cross-branch/state, and fanout-bound checks fail closed by construction and by test."),
        (R, "Selector predicate classes are compiler-declared, not learned; no learned selector-choice component exists to evaluate."),
        (S, "Budget-truncated scans return UNKNOWN and can never be forced into a committed singleton (mirrors DSH3-06)."),
        (U, "No serving-work comparison isolating selector construction cost exists."),
    )
    claims.append(
        AdvancedOperatorClaimV1(
            claim=AdvancedOperatorClaim.SELECTOR_CORRECTNESS,
            verdict=S,
            reason=(
                "DSH5-01/SLM-409's SelectorRefV1 contracts prove exact, permutation-invariant, "
                "fail-closed selector membership and freshness at the compiler-contract level. "
                "This is a runtime-correctness claim only; no learned-benefit or efficiency claim follows."
            ),
            dimension_verdicts=dv,
            dimension_reasons=dr,
            evidence_ids=("SLM-409.dsh5-01",),
        )
    )

    dv, dr = dims(
        (S, "Atomicity falls out of the unmodified OperatorLibraryV1._execute contract; rejection leaves state byte-identical."),
        (R, "No learned bulk-target-selection policy exists at this layer; selection is deterministic and caller-supplied."),
        (S, "Empty selectors and already-satisfied targets reject rather than silently no-op or partially commit."),
        (U, "Fanout/serving-cost comparison is DSH5-03's claim, not measured here."),
    )
    claims.append(
        AdvancedOperatorClaimV1(
            claim=AdvancedOperatorClaim.BULK_ATOMICITY,
            verdict=S,
            reason=(
                "DSH5-02/SLM-410's openui.map_set_property commits all-or-nothing over exact selector "
                "membership, with bulk/primitive-lowering equivalence and replay/permutation invariance "
                "proven by tests/test_dsl/test_bulk_operators.py. Runtime-correctness claim only."
            ),
            dimension_verdicts=dv,
            dimension_reasons=dr,
            evidence_ids=("SLM-410.dsh5-02",),
        )
    )

    dv, dr = dims(
        (F, "Fanout 1/2/4/8 legal-set/apply/replay/lowering checks are real but wiring evidence only."),
        (U, "DSH3-28's compatible typed-policy candidate is already rejected; no bulk-policy learned candidate exists."),
        (R, "No PARTIAL-coverage crossover stratum has been measured."),
        (U, "No matched five-arm (primitive_policy, oracle_primitive_sequence, bulk_policy, bulk_disabled, full_generation) serving comparison exists."),
    )
    claims.append(
        AdvancedOperatorClaimV1(
            claim=AdvancedOperatorClaim.CROSSOVER_WORK,
            verdict=U,
            reason=(
                "DSH5-03/SLM-411's own disposition is explicit: 'the report is unavailable, not a "
                "crossover result.' No fixture result substitutes for held-out meaningful parse or "
                "full serving-work measurement; the learned bulk-policy claim remains unadjudicated, "
                "not rejected and not supported."
            ),
            dimension_verdicts=dv,
            dimension_reasons=dr,
            evidence_ids=("SLM-411.dsh5-03",),
        )
    )

    dv, dr = dims(
        (S, "One-base preparation, exact read/write footprints, and atomic N-way composition/commit/replay are proven for disjoint and declared-commuting subsets, with a structural safety net for disagreeing declared-commuting pairs."),
        (R, "Neither DSH5-04 nor DSH5-05 wires a learned component; that is DSH5-06's separate claim."),
        (S, "Unknown/partial footprints, fanout-exceeded, and dependency cycles fail closed; never a silent partial commit."),
        (U, "No serving-work comparison for transaction commit exists."),
    )
    claims.append(
        AdvancedOperatorClaimV1(
            claim=AdvancedOperatorClaim.TRANSACTION_CONTRACTS_EXECUTION,
            verdict=S,
            reason=(
                "DSH5-04/SLM-412 (contracts) and DSH5-05/SLM-413 (execution) together prove atomic "
                "transaction composition, commit, and replay for the subset SLM-412 already proves "
                "preparable. Producer/consumer chaining across a newly-produced entity within one "
                "transaction remains explicitly unsupported. Runtime-correctness claim only."
            ),
            dimension_verdicts=dv,
            dimension_reasons=dr,
            evidence_ids=("SLM-412.dsh5-04", "SLM-413.dsh5-05"),
        )
    )

    dv, dr = dims(
        (S, "Bounded-K decode always re-verifies each ranked candidate with the real executor before commit; a deliberately corrupted conflict belief never commits a real conflict (test_shuffled_conflict_belief_never_commits_a_real_conflict)."),
        (R, "The N=2 hand-built fixture cannot support the issue's own 5%-causal-change / held-out-benefit criterion; DSH5-06 explicitly leaves it UNRUN, not asserted positive or negative."),
        (S, "PARTIAL coverage returns DEFER with zero model forwards; never force-emits a complete set."),
        (R, "Model-call counts are reported only for the 2-example fixture; not a powered systems comparison."),
    )
    claims.append(
        AdvancedOperatorClaimV1(
            claim=AdvancedOperatorClaim.SET_VALUED_SELECTION,
            verdict=R,
            reason=(
                "DSH5-06/SLM-414 delivers the safety-invariant wiring precondition for learned "
                "set-valued transaction selection, stress-tested adversarially, but the decision-"
                "relevant question -- does a learned score add held-out benefit over sequential/bulk "
                "policies -- has no corpus or baseline to answer. The runtime-safety result must not "
                "be read as satisfying the separate, unrun learned-benefit question."
            ),
            dimension_verdicts=dv,
            dimension_reasons=dr,
            evidence_ids=("SLM-414.dsh5-06",),
        )
    )

    dv, dr = dims(
        (S, "N-step conservative merge over verified branch histories is proven for AST_EDIT-only chains: buried conflicts, DELETE_MODIFY closure, stale/forged refs, and fresh-reference continuation all hold."),
        (R, "No learned component; this is a pure compiler-contract extension of the one-step merge bound."),
        (S, "Inexact (APPROXIMATE) coverage anywhere in either chain refuses the whole merge as UNSUPPORTED_EFFECT; a genuinely stale ref later in a sequence refuses without mutation."),
        (U, "No serving-work measurement for sequence merge exists."),
    )
    claims.append(
        AdvancedOperatorClaimV1(
            claim=AdvancedOperatorClaim.SEQUENCE_MERGE,
            verdict=S,
            reason=(
                "DSH5-07/SLM-415 extends the one-step conservative branch merge to N steps for the "
                "scope it actually implements (no TRANSACTION_COMMIT or mid-sequence FORK/COPY_STATE "
                "turns), preserving exact effect/read-write precision. Runtime-correctness claim only."
            ),
            dimension_verdicts=dv,
            dimension_reasons=dr,
            evidence_ids=("SLM-415.dsh5-07",),
        )
    )

    dv, dr = dims(
        (S, "The compiler-owned control-action legal set, typed dispatch, and deterministic baseline (singleton bypass + fixed priority) are proven correct and replay-exact by 12 tests in test_conversation_control_actions.py."),
        (R, "No learned control-policy head was trained; DSH5-08 explicitly defers training pending a first demonstrated held-out benefit -- this is the 'learning' half of the claim name, and it is unrun, not merely unreported."),
        (S, "Every control kind's legality is exactly enumerable in bounded time; MERGE_BRANCHES's one legitimate UNKNOWN (no authority_resolver) is proven distinct from a false UNSUPPORTED."),
        (U, "No serving-work measurement for control dispatch exists."),
    )
    claims.append(
        AdvancedOperatorClaimV1(
            claim=AdvancedOperatorClaim.CONTROL_PLANE_EXECUTION_LEARNING,
            verdict=S,
            reason=(
                "'Execution' and 'learning' are separated deliberately: the compiler-owned legal set, "
                "execution, and deterministic baseline (DSH5-08/SLM-416) are supported at the runtime-"
                "correctness axis; no learned control policy exists or was trained, so the 'learning' "
                "half remains unrun_conditional (see dimension_verdicts.learned_semantic_benefit). "
                "The headline verdict names only the supported execution half."
            ),
            dimension_verdicts=dv,
            dimension_reasons=dr,
            evidence_ids=("SLM-416.dsh5-08",),
        )
    )

    dv, dr = dims(
        (S, "The router admission preflight's own contract (unavailable modes correctly DEFER; no fixture is promoted to a quality/work claim) passes 2/2 AgentEvals criteria with zero execution errors."),
        (U, "No locked, group-disjoint, matched all-arm outcome corpus exists to train or evaluate a router; router regret is unmeasured."),
        (S, "Unavailable modes correctly DEFER rather than silently degrading to an approximate proxy."),
        (U, "No serving comparison exists."),
    )
    claims.append(
        AdvancedOperatorClaimV1(
            claim=AdvancedOperatorClaim.ADAPTIVE_ROUTING,
            verdict=U,
            reason=(
                "DSH5-09/SLM-417 is a real, executed preflight -- not an unattempted dimension -- and "
                "its own conclusion is 'unavailable': no matched group-disjoint corpus exists, so "
                "adaptive routing cannot be trained or evaluated at all yet. The preflight contract's "
                "own correctness (DEFER-on-unavailable) is separately supported and must not be read "
                "as router benefit."
            ),
            dimension_verdicts=dv,
            dimension_reasons=dr,
            evidence_ids=("SLM-417.dsh5-09",),
        )
    )

    dv, dr = dims(
        (S, "The delivered slice (edit-then-undo, undo-then-redo) extracts rows whose chosen/rejected actions are verified legal-set members and whose chosen_output_state matches independent replay."),
        (R, "All 7 of 7 named extraction patterns extract and replay-verify; merge conflict is honestly scoped out as a non-row legality constraint rather than a row. A bounded fixture-scale two-feature diagnostic scorer comparison now exists, but no SFT/preference training against a real DSH3 policy/control head, no powered corpus, and no certified held-out benefit measurement exist."),
        (R, "No CAP0/CAP1/CAP2 retention or calibration measurement exists yet for rows sourced from this module (the sixth slice's pairwise calibration proxy is a narrow, fixture-scale diagnostic, not a CAP-gated measurement)."),
        (R, "The turn-depth/context-view *structural* ablation dimension now exists (dsl.operators.replay_preference_context_views: five context-views x five turn-depths) and a bounded matched-variant comparison is wired (harness.preference.replay_preference_context_view_variants), but only over a tiny synthetic corpus and a two-feature linear scorer -- never a trained DSH3 policy/control head."),
    )
    claims.append(
        AdvancedOperatorClaimV1(
            claim=AdvancedOperatorClaim.EVENT_MEMORY,
            verdict=R,
            reason=(
                "DSH5-10/SLM-418's own doc states plainly: 'not yet dispositioned -- this PR delivers "
                "a scoped subset, not the full issue,' and that the issue 'should stay open.' The "
                "delivered row-extraction wiring is real and replay-verified, and a sixth slice adds a "
                "real (fixture-scale) turn-depth/context-view ablation and comparison, but the issue's "
                "held-out benefit question against a real trained policy remains unrun_conditional."
            ),
            dimension_verdicts=dv,
            dimension_reasons=dr,
            evidence_ids=("SLM-418.dsh5-10",),
        )
    )

    dv, dr = dims(
        (U, "No ParameterizedTemplateActionV1, parameter-slot lowering, or matched action matrix exists to evaluate."),
        (U, "No matched BASE/FLAT/CLOSED/PARAM semantic comparison exists."),
        (U, "No leakage-audited train-only manifest exists, so partial-coverage behavior cannot be assessed."),
        (U, "No matched serving/model arm exists."),
    )
    claims.append(
        AdvancedOperatorClaimV1(
            claim=AdvancedOperatorClaim.PARAMETERIZED_TEMPLATES,
            verdict=U,
            reason=(
                "DSH5-11/SLM-419's repository-wide tracked-path audit for TSA-013, TSA-014, "
                "ParameterizedTemplateAction, and template_manifest returned no matches. This is a "
                "real, completed audit with a confirmed-absent prerequisite, not an unattempted "
                "dimension: 'a fixture or fixed-span compression result is not evidence for typed "
                "open-leaf template actions and cannot supply the missing gate.' Absence of TSA "
                "artifacts is recorded as UNAVAILABLE, never as an assumed benefit or failure."
            ),
            dimension_verdicts=dv,
            dimension_reasons=dr,
            evidence_ids=("SLM-419.dsh5-11",),
        )
    )

    dv, dr = dims(
        (U, "No systems harness reached completion for any DSH5 abstraction (bulk, transaction, control, or router)."),
        (U, "Serving efficiency is a distinct axis from semantic benefit by construction; no semantic-gain result substitutes for it."),
        (U, "No PARTIAL-coverage systems stratum has been measured for any abstraction."),
        (U, "DSH5-03's own systems contract is 'unavailable, not a crossover result,' consistent with the inherited DSH3-33/SLM-407 'no compatible five-arm measured serving comparison' finding."),
    )
    claims.append(
        AdvancedOperatorClaimV1(
            claim=AdvancedOperatorClaim.SYSTEMS_EFFICIENCY,
            verdict=U,
            reason=(
                "No DSH5 abstraction has a compatible matched multi-arm serving-work comparison. A "
                "semantic gain (none measured) cannot establish efficiency and a latency gain (none "
                "measured) cannot establish semantics; both remain unavailable independently."
            ),
            dimension_verdicts=dv,
            dimension_reasons=dr,
            evidence_ids=("SLM-411.dsh5-03", "SLM-408.dsh3-33"),
        )
    )

    inherited_policy = InheritedPolicyInventoryV1(
        source_disposition="SLM-408 (DSH3-33)",
        source_artifact_path="docs/design/dsh3-33-rebased-cap2-disposition-20260726-local/report.json",
        dsh5_may_start=bool(dsh5_block["may_start"]),
        allowed_heads=tuple(dsh5_block["allowed_heads"]),
        allowed_objectives=tuple(dsh5_block["allowed_objectives"]),
        allowed_actions=tuple(dsh5_block["allowed_actions"]),
        reason=str(dsh5_block["reason"]),
    )

    retention = (
        RetentionClaimV1(
            capability="CAP0",
            verdict=R,
            reason="No learned CAP2/DSH5 candidate reached a retention evaluation; inherited unchanged from SLM-408 (DSH3-33).",
            evidence_ids=("SLM-408.dsh3-33",),
        ),
        RetentionClaimV1(
            capability="CAP1",
            verdict=U,
            reason="No CAP1-compatible learned candidate or retention artifact exists; inherited unchanged from SLM-408 (DSH3-33).",
            evidence_ids=("SLM-408.dsh3-33",),
        ),
        RetentionClaimV1(
            capability="CAP2",
            verdict=S,
            reason=(
                "CERT_CAP2 remains not-issued per SLM-408 (DSH3-33). DSH5's supported claims -- "
                "selector correctness, bulk atomicity, transaction contracts/execution, sequence "
                "merge, and control-plane execution -- are compiler-owned runtime utilities, never a "
                "learned CAP2 capability, so the certificate posture is retained unchanged, not "
                "advanced."
            ),
            evidence_ids=(
                "SLM-408.dsh3-33",
                "SLM-409.dsh5-01",
                "SLM-410.dsh5-02",
                "SLM-412.dsh5-04",
                "SLM-413.dsh5-05",
                "SLM-415.dsh5-07",
                "SLM-416.dsh5-08",
            ),
        ),
    )

    return AdvancedOperatorDispositionV1(
        evidence=evidence,
        claims=tuple(claims),
        inherited_policy=inherited_policy,
        retention=retention,
        advanced_path_enabled_by_default=False,
        recommendation="retain_as_compiler_utility",
        recommendation_reason=(
            "Selector correctness, bulk atomicity, transaction contracts/execution, sequence merge, "
            "and control-plane execution are proven runtime-correctness compiler utilities and should "
            "be retained as such. Crossover work, adaptive routing, parameterized templates, and "
            "systems efficiency are UNAVAILABLE (their measurement prerequisites do not exist); "
            "set-valued selection and event memory are UNRUN_CONDITIONAL wiring preconditions whose "
            "held-out-benefit question remains open and may continue as default-off research, but "
            "none is promoted, shipped, or enabled by default from this disposition alone."
        ),
        historical_disposition="docs/design/dsh3-33-rebased-cap2-disposition-20260726-local/report.json (historical, immutable)",
        version_stamp=version_stamp,
    )


__all__ = [
    "ALLOWED_RECOMMENDATIONS",
    "COMPONENT_BY_ISSUE",
    "AdvancedOperatorClaim",
    "AdvancedOperatorClaimV1",
    "AdvancedOperatorDimension",
    "AdvancedOperatorDispositionV1",
    "AdvancedOperatorEvidenceV1",
    "AdvancedOperatorVerdict",
    "InheritedPolicyInventoryV1",
    "RetentionClaimV1",
    "build_advanced_operator_disposition",
    "component_version_stamps",
]
