"""Strict revmath task / corpus / result / repair / report schemas (HARN-02).

Fail closed: malformed or underspecified payloads raise. Results embed
``SolverJudgmentV1`` (KERN-01) — never a parallel outcome enum. Repair records
may only touch knobs in ``REVMATH_MUTABLE_REPAIR_KNOBS``; proposition /
assumption / campaign identity are immutable by construction.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from slm_training.formal.judgment import JUDGMENT_SCHEMA, SolverJudgmentV1
from slm_training.harness_core.lineage.records import canonical_json, content_sha

RevmathTaskKind = Literal[
    "assumption_ablation",
    "forward_theorem",
    "reversal",
    "constructivization",
    "computable_finite_counterexample",
    "quantitative_bound_extraction",
    "computability_classification",
]

TheoremDirection = Literal[
    "forward",
    "reversal",
    "equivalence",
    "constructive_variant",
]

InterpretationStatus = Literal[
    "explicit_reversal",
    "explicit_interpretation",
    "uninterpreted",
    "not_applicable",
]

FourAxisName = Literal[
    "assumption_strength",
    "computability",
    "resource_bounds",
    "implementation_refinement",
]

AxisClaimStatus = Literal[
    "proved_axis",
    "refuted_axis",
    "unknown_axis",
    "empirical_remainder",
    "not_claimed",
]

# Mirrors revmath_harness_parity.json self_healing.may_modify (parity-tested).
REVMATH_MUTABLE_REPAIR_KNOBS: frozenset[str] = frozenset(
    {
        "tactic_sequence",
        "proof_search_strategy",
        "retrieval_choices",
        "type_equivalent_theorem_selection",
        "serialization_bridge_generation",
        "temporary_decomposition",
        "resource_allocation_within_locked_budget",
    }
)

# Simpson-style subsystems: labels require explicit interpretation/reversal.
_KNOWN_RM_SUBSYSTEMS: frozenset[str] = frozenset(
    {"RCA0", "WKL0", "ACA0", "ATR0", "Pi11-CA0", "RCA", "WKL", "ACA", "ATR"}
)

_HEX64 = r"^[0-9a-f]{64}$"
_ID = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"

REVMATH_SCHEMA_VERSIONS: Mapping[str, frozenset[str]] = {
    "task": frozenset({"revmath_task/v1"}),
    "corpus_entry": frozenset({"revmath_corpus_entry/v1"}),
    "result": frozenset({"revmath_result/v1"}),
    "repair": frozenset({"revmath_repair/v1"}),
    "report": frozenset({"revmath_report/v1"}),
    "four_axis": frozenset({"revmath_four_axis/v1"}),
}



class RevmathSchemaError(ValueError):
    """Malformed, underspecified, or identity-violating revmath payload."""


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def canonical_hash(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    elif hasattr(value, "to_dict"):
        value = value.to_dict()
    return content_sha(value)


class RevmathPropositionIdentityV1(StrictModel):
    """Frozen proposition identity (must_freeze: proposition)."""

    proposition_id: str = Field(pattern=_ID)
    statement: str = Field(min_length=8)
    statement_sha256: str = Field(pattern=_HEX64)

    @model_validator(mode="after")
    def _statement_digest_matches(self) -> RevmathPropositionIdentityV1:
        digest = content_sha(self.statement)
        if digest != self.statement_sha256:
            raise ValueError(
                "statement_sha256 must equal content_sha(statement); "
                "proposition identity is fail-closed"
            )
        return self


class RevmathBaseTheoryV1(StrictModel):
    """Base theory + allowed assumptions (must_freeze)."""

    base_theory_id: str = Field(pattern=_ID, min_length=2)
    allowed_assumption_ids: tuple[str, ...] = ()
    interpretation_status: InterpretationStatus

    @model_validator(mode="after")
    def _rm_labels_need_interpretation(self) -> RevmathBaseTheoryV1:
        if self.base_theory_id in _KNOWN_RM_SUBSYSTEMS and self.interpretation_status in (
            "uninterpreted",
            "not_applicable",
        ):
            raise ValueError(
                f"base theory {self.base_theory_id!r} requires "
                "explicit_reversal or explicit_interpretation status "
                "(never inferred from axiom prints)"
            )
        return self


class RevmathCorpusIdentityV1(StrictModel):
    corpus_id: str = Field(pattern=_ID)
    corpus_sha256: str = Field(pattern=_HEX64)
    entry_id: str = Field(pattern=_ID)


class RevmathBudgetV1(StrictModel):
    max_wall_seconds: float = Field(gt=0, le=3600)
    max_tactic_steps: int = Field(ge=1, le=1_000_000)
    max_memory_mb: int | None = Field(default=None, ge=1)
    stopping_rule_ids: tuple[str, ...] = Field(min_length=1)


class RevmathVerifierJudgeIdentityV1(StrictModel):
    """Verifier + judge identity; judgment schema is locked to KERN-01."""

    verifier_id: str = Field(pattern=_ID)
    judge_id: str = Field(pattern=_ID)
    judgment_schema: Literal["solver_judgment/v1"] = JUDGMENT_SCHEMA


class RevmathCampaignBindingV1(StrictModel):
    campaign_id: str = Field(pattern=_ID)
    experiment_id: str = Field(pattern=_ID)
    campaign_manifest_sha256: str = Field(pattern=_HEX64)
    locked_arm_ids: tuple[str, ...] = Field(min_length=1)


class RevmathTaskV1(StrictModel):
    """Strict revmath task; all freeze fields are required."""

    schema_version: Literal["revmath_task/v1"] = "revmath_task/v1"
    task_id: str = Field(pattern=_ID)
    task_kind: RevmathTaskKind
    proposition: RevmathPropositionIdentityV1
    base_theory: RevmathBaseTheoryV1
    theorem_direction: TheoremDirection
    corpus: RevmathCorpusIdentityV1
    budget: RevmathBudgetV1
    verifier_judge: RevmathVerifierJudgeIdentityV1
    campaign: RevmathCampaignBindingV1

    def identity_digest(self) -> str:
        """Digest over identity fields only (repair-stable)."""
        return content_sha(
            {
                "proposition_id": self.proposition.proposition_id,
                "statement_sha256": self.proposition.statement_sha256,
                "base_theory_id": self.base_theory.base_theory_id,
                "allowed_assumption_ids": list(self.base_theory.allowed_assumption_ids),
                "interpretation_status": self.base_theory.interpretation_status,
                "theorem_direction": self.theorem_direction,
                "corpus_id": self.corpus.corpus_id,
                "corpus_sha256": self.corpus.corpus_sha256,
                "entry_id": self.corpus.entry_id,
                "campaign_manifest_sha256": self.campaign.campaign_manifest_sha256,
                "campaign_id": self.campaign.campaign_id,
                "experiment_id": self.campaign.experiment_id,
                "locked_arm_ids": list(self.campaign.locked_arm_ids),
                "budget": self.budget.model_dump(mode="json"),
                "verifier_id": self.verifier_judge.verifier_id,
                "judge_id": self.verifier_judge.judge_id,
                "judgment_schema": self.verifier_judge.judgment_schema,
                "task_kind": self.task_kind,
            }
        )

    def canonical_hash(self) -> str:
        return canonical_hash(self)


class RevmathCorpusEntryV1(StrictModel):
    schema_version: Literal["revmath_corpus_entry/v1"] = "revmath_corpus_entry/v1"
    identity: RevmathCorpusIdentityV1
    task: RevmathTaskV1

    @model_validator(mode="after")
    def _task_corpus_matches(self) -> RevmathCorpusEntryV1:
        if self.task.corpus != self.identity:
            raise ValueError("corpus entry identity must equal task.corpus")
        return self


class AxisCertificateRefV1(StrictModel):
    axis: FourAxisName
    status: AxisClaimStatus
    certificate_sha256: str | None = Field(default=None, pattern=_HEX64)
    theorem_ref: str | None = Field(default=None, min_length=1)
    bound_ast_id: str | None = Field(default=None, pattern=_ID)
    notes: str = ""

    @model_validator(mode="after")
    def _certs_required_for_decisions(self) -> AxisCertificateRefV1:
        if self.status in ("proved_axis", "refuted_axis"):
            if not self.certificate_sha256:
                raise ValueError(
                    f"axis {self.axis!r} status {self.status!r} requires certificate_sha256"
                )
            if not self.theorem_ref:
                raise ValueError(
                    f"axis {self.axis!r} status {self.status!r} requires theorem_ref"
                )
        if (
            self.axis == "resource_bounds"
            and self.status in ("proved_axis", "refuted_axis")
            and not self.bound_ast_id
        ):
            raise ValueError(
                "resource_bounds proved/refuted claims require bound_ast_id "
                "(EVID-04 safe AST id; no raw eval)"
            )
        return self


class RevmathFourAxisAnalysisV1(StrictModel):
    """Four independent claim axes carried by results.

    Ledger field owners land on FormalPreflightV1 under EVID-03; this type is
    the result attachment, not a parallel evidence store.
    """

    schema_version: Literal["revmath_four_axis/v1"] = "revmath_four_axis/v1"
    assumption_strength: AxisCertificateRefV1
    computability: AxisCertificateRefV1
    resource_bounds: AxisCertificateRefV1
    implementation_refinement: AxisCertificateRefV1
    formal_preflight_sha256: str | None = Field(default=None, pattern=_HEX64)

    @model_validator(mode="after")
    def _axis_slots_match(self) -> RevmathFourAxisAnalysisV1:
        expected = (
            ("assumption_strength", self.assumption_strength),
            ("computability", self.computability),
            ("resource_bounds", self.resource_bounds),
            ("implementation_refinement", self.implementation_refinement),
        )
        for name, axis in expected:
            if axis.axis != name:
                raise ValueError(
                    f"four-axis slot {name!r} carries axis={axis.axis!r}; fail closed"
                )
        return self


class RevmathProofArtifactRefV1(StrictModel):
    proof_sha256: str = Field(pattern=_HEX64)
    checker_id: str = Field(pattern=_ID)
    checker_report_sha256: str | None = Field(default=None, pattern=_HEX64)


class RevmathResultV1(StrictModel):
    """Task result: KERN-01 judgment + proof refs + four-axis analysis."""

    schema_version: Literal["revmath_result/v1"] = "revmath_result/v1"
    result_id: str = Field(pattern=_ID)
    task_id: str = Field(pattern=_ID)
    task_identity_digest: str = Field(pattern=_HEX64)
    judgment: dict[str, Any]
    proof: RevmathProofArtifactRefV1 | None = None
    four_axis: RevmathFourAxisAnalysisV1

    @model_validator(mode="after")
    def _judgment_is_solver_judgment(self) -> RevmathResultV1:
        try:
            judgment = SolverJudgmentV1.from_dict(self.judgment)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "revmath results require SolverJudgmentV1 "
                f"(schema {JUDGMENT_SCHEMA!r}); cannot normalize unknown payloads "
                f"into proof success/failure ({exc})"
            ) from exc
        if judgment.outcome.value in ("witnessed", "refuted") and self.proof is None:
            raise ValueError(
                "witnessed/refuted results require proof artifact refs "
                "(fail closed; unknown stays unknown)"
            )
        canonical = judgment.to_dict()
        if self.judgment != canonical:
            return self.model_copy(update={"judgment": canonical})
        return self

    def solver_judgment(self) -> SolverJudgmentV1:
        return SolverJudgmentV1.from_dict(self.judgment)


class RevmathRepairKnobChangeV1(StrictModel):
    knob: str = Field(min_length=1)
    before_value_sha256: str = Field(pattern=_HEX64)
    after_value_sha256: str = Field(pattern=_HEX64)

    @model_validator(mode="after")
    def _knob_is_mutable(self) -> RevmathRepairKnobChangeV1:
        if self.knob not in REVMATH_MUTABLE_REPAIR_KNOBS:
            raise ValueError(
                f"repair knob {self.knob!r} is not mutable; "
                f"allowed={sorted(REVMATH_MUTABLE_REPAIR_KNOBS)}"
            )
        if self.before_value_sha256 == self.after_value_sha256:
            raise ValueError(f"repair knob {self.knob!r} records no change")
        return self


class RevmathRepairRecordV1(StrictModel):
    """Proposition-preserving repair: immutable identity + mutable knobs only."""

    schema_version: Literal["revmath_repair/v1"] = "revmath_repair/v1"
    repair_id: str = Field(pattern=_ID)
    task_identity_digest: str = Field(pattern=_HEX64)
    proposition_id: str = Field(pattern=_ID)
    statement_sha256: str = Field(pattern=_HEX64)
    base_theory_id: str = Field(pattern=_ID)
    allowed_assumption_ids: tuple[str, ...] = ()
    campaign_manifest_sha256: str = Field(pattern=_HEX64)
    campaign_id: str = Field(pattern=_ID)
    experiment_id: str = Field(pattern=_ID)
    before_proof_sha256: str = Field(pattern=_HEX64)
    after_proof_sha256: str = Field(pattern=_HEX64)
    knob_changes: tuple[RevmathRepairKnobChangeV1, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _proofs_differ(self) -> RevmathRepairRecordV1:
        if self.before_proof_sha256 == self.after_proof_sha256:
            raise ValueError("repair must record distinct before/after proof artifacts")
        return self


class RevmathReportV1(StrictModel):
    schema_version: Literal["revmath_report/v1"] = "revmath_report/v1"
    report_id: str = Field(pattern=_ID)
    corpus_id: str = Field(pattern=_ID)
    campaign_manifest_sha256: str = Field(pattern=_HEX64)
    task_ids: tuple[str, ...] = Field(min_length=1)
    result_ids: tuple[str, ...] = ()
    repair_ids: tuple[str, ...] = ()
    judgment_counts: dict[str, int] = Field(default_factory=dict)
    version_stamp: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _judgment_counts_closed(self) -> RevmathReportV1:
        allowed = {"witnessed", "refuted", "unknown", "invalid"}
        bad = set(self.judgment_counts) - allowed
        if bad:
            raise ValueError(
                f"judgment_counts keys {sorted(bad)} are not SolverJudgmentV1 "
                "outcomes; refuse success/failure laundering"
            )
        for key, value in self.judgment_counts.items():
            if value < 0:
                raise ValueError(f"judgment_counts[{key!r}] must be >= 0")
        return self


def _require_schema(kind: str, data: Mapping[str, Any]) -> str:
    if not isinstance(data, Mapping):
        raise RevmathSchemaError(f"{kind} payload must be a JSON object")
    schema = data.get("schema_version")
    allowed = REVMATH_SCHEMA_VERSIONS[kind]
    if schema not in allowed:
        raise RevmathSchemaError(
            f"unsupported {kind} schema_version {schema!r}; "
            f"expected one of {sorted(allowed)} (fail closed, no soft migration)"
        )
    return str(schema)


def _parse[T: StrictModel](model: type[T], kind: str, data: Mapping[str, Any]) -> T:
    _require_schema(kind, data)
    try:
        return model.model_validate(dict(data))
    except Exception as exc:
        raise RevmathSchemaError(f"invalid {kind}: {exc}") from exc


def parse_revmath_task(data: Mapping[str, Any]) -> RevmathTaskV1:
    return _parse(RevmathTaskV1, "task", data)


def parse_revmath_corpus_entry(data: Mapping[str, Any]) -> RevmathCorpusEntryV1:
    return _parse(RevmathCorpusEntryV1, "corpus_entry", data)


def parse_revmath_result(data: Mapping[str, Any]) -> RevmathResultV1:
    return _parse(RevmathResultV1, "result", data)


def parse_revmath_repair(data: Mapping[str, Any]) -> RevmathRepairRecordV1:
    return _parse(RevmathRepairRecordV1, "repair", data)


def parse_revmath_report(data: Mapping[str, Any]) -> RevmathReportV1:
    return _parse(RevmathReportV1, "report", data)


def assert_repair_preserves_identity(
    task: RevmathTaskV1, repair: RevmathRepairRecordV1
) -> None:
    """Fail closed if a repair alters proposition / assumption / campaign identity."""
    expected = task.identity_digest()
    if repair.task_identity_digest != expected:
        raise RevmathSchemaError(
            "repair.task_identity_digest does not match task identity digest"
        )
    if repair.proposition_id != task.proposition.proposition_id:
        raise RevmathSchemaError("repair cannot alter proposition_id")
    if repair.statement_sha256 != task.proposition.statement_sha256:
        raise RevmathSchemaError("repair cannot alter proposition statement")
    if repair.base_theory_id != task.base_theory.base_theory_id:
        raise RevmathSchemaError("repair cannot alter base_theory_id")
    if tuple(repair.allowed_assumption_ids) != tuple(
        task.base_theory.allowed_assumption_ids
    ):
        raise RevmathSchemaError("repair cannot alter allowed_assumption_ids")
    if repair.campaign_manifest_sha256 != task.campaign.campaign_manifest_sha256:
        raise RevmathSchemaError("repair cannot alter campaign_manifest_sha256")
    if repair.campaign_id != task.campaign.campaign_id:
        raise RevmathSchemaError("repair cannot alter campaign_id")
    if repair.experiment_id != task.campaign.experiment_id:
        raise RevmathSchemaError("repair cannot alter experiment_id")


def export_json_schemas() -> dict[str, dict[str, Any]]:
    """JSON Schema drafts for each public contract (validation / docs)."""
    return {
        "revmath_task/v1": RevmathTaskV1.model_json_schema(),
        "revmath_corpus_entry/v1": RevmathCorpusEntryV1.model_json_schema(),
        "revmath_result/v1": RevmathResultV1.model_json_schema(),
        "revmath_repair/v1": RevmathRepairRecordV1.model_json_schema(),
        "revmath_report/v1": RevmathReportV1.model_json_schema(),
        "revmath_four_axis/v1": RevmathFourAxisAnalysisV1.model_json_schema(),
    }


def dump_canonical(model: BaseModel) -> str:
    return canonical_json(model.model_dump(mode="json"))


__all__ = [
    "REVMATH_MUTABLE_REPAIR_KNOBS",
    "REVMATH_SCHEMA_VERSIONS",
    "AxisCertificateRefV1",
    "AxisClaimStatus",
    "FourAxisName",
    "InterpretationStatus",
    "RevmathBaseTheoryV1",
    "RevmathBudgetV1",
    "RevmathCampaignBindingV1",
    "RevmathCorpusEntryV1",
    "RevmathCorpusIdentityV1",
    "RevmathFourAxisAnalysisV1",
    "RevmathProofArtifactRefV1",
    "RevmathPropositionIdentityV1",
    "RevmathRepairKnobChangeV1",
    "RevmathRepairRecordV1",
    "RevmathReportV1",
    "RevmathResultV1",
    "RevmathSchemaError",
    "RevmathTaskKind",
    "RevmathTaskV1",
    "RevmathVerifierJudgeIdentityV1",
    "TheoremDirection",
    "assert_repair_preserves_identity",
    "canonical_hash",
    "dump_canonical",
    "export_json_schemas",
    "parse_revmath_corpus_entry",
    "parse_revmath_repair",
    "parse_revmath_report",
    "parse_revmath_result",
    "parse_revmath_task",
]
