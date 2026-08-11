"""Constructivization generation + validation (HARN-06 / SLM-546).

Replace qualitative/nonconstructive claims with explicit finite witnesses or
bounds where possible. Modes:

* ``bounded_finite_analogue`` — classical claim → bounded/finite variant
* ``explicit_witness_producing`` — formulation that must produce a witness
* ``oracle_relative`` — formulation relative to a named oracle
* ``documented_nonconstructive_remainder`` — records what stays nonconstructive

Every constructivized claim states what was weakened/bounded and must not
masquerade as the original theorem (distinct statement digest; task proposition
is the constructivized statement only).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from slm_training.harness_core.lineage.records import content_sha
from slm_training.harnesses.reasoning.revmath.schemas import (
    RevmathSchemaError,
    RevmathTaskV1,
)

CONSTRUCTIVIZATION_META_SCHEMA = "revmath_constructivization_meta/v1"
CONSTRUCTIVIZATION_VARIANT_SCHEMA = "revmath_constructivization_variant/v1"
CONSTRUCTIVIZATION_REPORT_SCHEMA = "revmath_constructivization_report/v1"

ConstructivizationMode = Literal[
    "bounded_finite_analogue",
    "explicit_witness_producing",
    "oracle_relative",
    "documented_nonconstructive_remainder",
]
HermeticConstructivizationScenario = Literal[
    "bounded_ok",
    "witness_ok",
    "oracle_ok",
    "remainder_ok",
    "masquerade",
    "timeout",
    "incomplete",
]
ConstructivizationOutcomeName = Literal["witnessed", "refuted", "unknown", "invalid"]

_MODES: frozenset[str] = frozenset(
    {
        "bounded_finite_analogue",
        "explicit_witness_producing",
        "oracle_relative",
        "documented_nonconstructive_remainder",
    }
)


def _require_statement_digest(statement: str, digest: str, *, label: str) -> None:
    actual = content_sha(statement)
    if actual != digest:
        raise RevmathSchemaError(
            f"{label} statement_sha256 mismatch "
            f"(expected {digest!r}, got {actual!r})"
        )


@dataclass(frozen=True)
class ConstructivizationMetaV1:
    """Frozen constructivization sidecar (not part of RevmathTaskV1 identity)."""

    schema_version: str
    meta_id: str
    mode: ConstructivizationMode
    original_statement: str
    original_statement_sha256: str
    constructivized_statement: str
    constructivized_statement_sha256: str
    weakening_description: str
    hermetic_scenario: HermeticConstructivizationScenario
    # EVID-04 bound AST id placeholder (required for bounded_finite_analogue).
    bound_ast_id: str | None = None
    witness_spec: str | None = None
    oracle_id: str | None = None
    remainder_note: str | None = None
    oracle_status: str = "ok"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "meta_id": self.meta_id,
            "mode": self.mode,
            "original_statement": self.original_statement,
            "original_statement_sha256": self.original_statement_sha256,
            "constructivized_statement": self.constructivized_statement,
            "constructivized_statement_sha256": self.constructivized_statement_sha256,
            "weakening_description": self.weakening_description,
            "hermetic_scenario": self.hermetic_scenario,
            "bound_ast_id": self.bound_ast_id,
            "witness_spec": self.witness_spec,
            "oracle_id": self.oracle_id,
            "remainder_note": self.remainder_note,
            "oracle_status": self.oracle_status,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ConstructivizationMetaV1:
        if data.get("schema_version") != CONSTRUCTIVIZATION_META_SCHEMA:
            raise RevmathSchemaError(
                f"expected schema {CONSTRUCTIVIZATION_META_SCHEMA!r}, "
                f"got {data.get('schema_version')!r}"
            )
        mode = str(data.get("mode", ""))
        if mode not in _MODES:
            raise RevmathSchemaError(f"unknown constructivization mode {mode!r}")
        scenario = str(data.get("hermetic_scenario", ""))
        allowed = (
            "bounded_ok",
            "witness_ok",
            "oracle_ok",
            "remainder_ok",
            "masquerade",
            "timeout",
            "incomplete",
        )
        if scenario not in allowed:
            raise RevmathSchemaError(f"unknown hermetic_scenario {scenario!r}")
        original = str(data["original_statement"])
        original_sha = str(data["original_statement_sha256"])
        constructivized = str(data["constructivized_statement"])
        constructivized_sha = str(data["constructivized_statement_sha256"])
        _require_statement_digest(original, original_sha, label="original")
        _require_statement_digest(
            constructivized, constructivized_sha, label="constructivized"
        )
        weakening = str(data.get("weakening_description") or "").strip()
        if not weakening:
            raise RevmathSchemaError(
                "weakening_description must state what was weakened/bounded"
            )
        bound_ast_id = data.get("bound_ast_id")
        witness_spec = data.get("witness_spec")
        oracle_id = data.get("oracle_id")
        remainder_note = data.get("remainder_note")
        if mode == "bounded_finite_analogue" and (
            not isinstance(bound_ast_id, str) or not bound_ast_id.startswith("bound.")
        ):
            raise RevmathSchemaError(
                "bounded_finite_analogue requires bound_ast_id starting with "
                "'bound.' (EVID-04 placeholder; no raw eval)"
            )
        if mode == "explicit_witness_producing" and not (
            isinstance(witness_spec, str) and witness_spec.strip()
        ):
            raise RevmathSchemaError(
                "explicit_witness_producing requires non-empty witness_spec"
            )
        if mode == "oracle_relative" and not (
            isinstance(oracle_id, str) and oracle_id.strip()
        ):
            raise RevmathSchemaError("oracle_relative requires non-empty oracle_id")
        if mode == "documented_nonconstructive_remainder" and not (
            isinstance(remainder_note, str) and remainder_note.strip()
        ):
            raise RevmathSchemaError(
                "documented_nonconstructive_remainder requires remainder_note"
            )
        return cls(
            schema_version=CONSTRUCTIVIZATION_META_SCHEMA,
            meta_id=str(data["meta_id"]),
            mode=mode,  # type: ignore[arg-type]
            original_statement=original,
            original_statement_sha256=original_sha,
            constructivized_statement=constructivized,
            constructivized_statement_sha256=constructivized_sha,
            weakening_description=weakening,
            hermetic_scenario=scenario,  # type: ignore[arg-type]
            bound_ast_id=str(bound_ast_id) if bound_ast_id is not None else None,
            witness_spec=str(witness_spec) if witness_spec is not None else None,
            oracle_id=str(oracle_id) if oracle_id is not None else None,
            remainder_note=str(remainder_note) if remainder_note is not None else None,
            oracle_status=str(data.get("oracle_status") or "ok"),
        )


@dataclass(frozen=True)
class ConstructivizationVariantV1:
    """One constructivized formulation — never identical to the original theorem."""

    schema_version: str
    variant_id: str
    mode: ConstructivizationMode
    original_statement_sha256: str
    constructivized_statement_sha256: str
    weakening_description: str
    bound_ast_id: str | None
    variant_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "variant_id": self.variant_id,
            "mode": self.mode,
            "original_statement_sha256": self.original_statement_sha256,
            "constructivized_statement_sha256": self.constructivized_statement_sha256,
            "weakening_description": self.weakening_description,
            "bound_ast_id": self.bound_ast_id,
            "variant_digest": self.variant_digest,
            "masquerades_as_original": False,
        }


@dataclass(frozen=True)
class ConstructivizationReportV1:
    """Report for one constructivization check (weakening always explicit)."""

    schema_version: str
    task_id: str
    meta_id: str
    variant: ConstructivizationVariantV1
    outcome: ConstructivizationOutcomeName
    judgment: Mapping[str, Any]
    weakening_description: str
    original_statement_sha256: str
    constructivized_statement_sha256: str
    report_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "meta_id": self.meta_id,
            "variant": self.variant.to_dict(),
            "outcome": self.outcome,
            "judgment": dict(self.judgment),
            "weakening_description": self.weakening_description,
            "original_statement_sha256": self.original_statement_sha256,
            "constructivized_statement_sha256": self.constructivized_statement_sha256,
            "masquerades_as_original": False,
            "report_digest": self.report_digest,
        }


def parse_constructivization_meta(data: Mapping[str, Any]) -> ConstructivizationMetaV1:
    return ConstructivizationMetaV1.from_dict(data)


def assert_not_masquerading(meta: ConstructivizationMetaV1) -> None:
    """Fail closed when constructivized digest equals the original theorem."""

    if meta.constructivized_statement_sha256 == meta.original_statement_sha256:
        raise RevmathSchemaError(
            "constructivized statement must not masquerade as the original "
            "theorem (statement_sha256 collision); record a distinct bounded/"
            "witness/oracle/remainder formulation"
        )
    if meta.constructivized_statement.strip() == meta.original_statement.strip():
        raise RevmathSchemaError(
            "constructivized statement text equals original; weakening must "
            "change the claim"
        )


def assert_meta_matches_task(
    task: RevmathTaskV1, meta: ConstructivizationMetaV1
) -> None:
    """Task proposition must be the constructivized claim, never the original."""

    assert_not_masquerading(meta)
    if task.proposition.statement_sha256 != meta.constructivized_statement_sha256:
        raise RevmathSchemaError(
            "task proposition must be the constructivized statement "
            f"(task={task.proposition.statement_sha256!r} "
            f"constructivized={meta.constructivized_statement_sha256!r}); "
            "never attach the original classical theorem digest"
        )
    if task.proposition.statement != meta.constructivized_statement:
        raise RevmathSchemaError(
            "task proposition.statement must equal constructivized_statement"
        )


def generate_constructivization_variant(
    task: RevmathTaskV1,
    meta: ConstructivizationMetaV1,
) -> ConstructivizationVariantV1:
    """Materialize the single constructivized variant (deterministic)."""

    assert_meta_matches_task(task, meta)
    body = {
        "mode": meta.mode,
        "original_statement_sha256": meta.original_statement_sha256,
        "constructivized_statement_sha256": meta.constructivized_statement_sha256,
        "weakening_description": meta.weakening_description,
        "bound_ast_id": meta.bound_ast_id,
        "task_kind": "constructivization",
        "meta_id": meta.meta_id,
    }
    digest = content_sha(body)
    return ConstructivizationVariantV1(
        schema_version=CONSTRUCTIVIZATION_VARIANT_SCHEMA,
        variant_id=f"{task.task_id}.variant.{meta.mode}",
        mode=meta.mode,
        original_statement_sha256=meta.original_statement_sha256,
        constructivized_statement_sha256=meta.constructivized_statement_sha256,
        weakening_description=meta.weakening_description,
        bound_ast_id=meta.bound_ast_id,
        variant_digest=digest,
    )


def materialize_constructivized_task(
    task: RevmathTaskV1,
    meta: ConstructivizationMetaV1,
) -> RevmathTaskV1:
    """Clone task so proposition is exactly the constructivized statement."""

    assert_not_masquerading(meta)
    return task.model_copy(
        update={
            "task_id": f"{task.task_id}.constructivized",
            "task_kind": "constructivization",
            "proposition": task.proposition.model_copy(
                update={
                    "statement": meta.constructivized_statement,
                    "statement_sha256": meta.constructivized_statement_sha256,
                }
            ),
        }
    )


def _outcome_name(judgment: Mapping[str, Any]) -> ConstructivizationOutcomeName:
    raw = str(judgment.get("outcome", "unknown"))
    if raw in ("witnessed", "refuted", "unknown", "invalid"):
        return raw  # type: ignore[return-value]
    return "unknown"


def evaluate_constructivization(
    task: RevmathTaskV1,
    meta: ConstructivizationMetaV1,
    *,
    run_variant: Callable[[RevmathTaskV1, ConstructivizationVariantV1], Any],
) -> ConstructivizationReportV1:
    """Validate one constructivized variant via the shared runner.

    ``run_variant`` must reuse ``run_revmath_task`` (no second orchestrator).
    """

    assert_meta_matches_task(task, meta)
    if task.task_kind != "constructivization":
        raise RevmathSchemaError(
            f"evaluate_constructivization requires task_kind=constructivization, "
            f"got {task.task_kind!r}"
        )
    variant = generate_constructivization_variant(task, meta)
    record = run_variant(task, variant)
    judgment = record.result.solver_judgment().to_dict()
    # Masquerade scenario from hermetic checker → invalid even if checker said ok.
    if meta.hermetic_scenario == "masquerade":
        judgment = {
            "schema": judgment.get("schema", "solver_judgment/v1"),
            "outcome": "invalid",
            "payload": {
                "kind": "invalid_reason",
                "reason": "payload_mismatch",
            },
            "checked": True,
        }
    outcome = _outcome_name(judgment)
    payload = {
        "schema_version": CONSTRUCTIVIZATION_REPORT_SCHEMA,
        "task_id": task.task_id,
        "meta_id": meta.meta_id,
        "variant": variant.to_dict(),
        "outcome": outcome,
        "judgment": judgment,
        "weakening_description": meta.weakening_description,
        "original_statement_sha256": meta.original_statement_sha256,
        "constructivized_statement_sha256": meta.constructivized_statement_sha256,
        "masquerades_as_original": False,
    }
    return ConstructivizationReportV1(
        schema_version=CONSTRUCTIVIZATION_REPORT_SCHEMA,
        task_id=task.task_id,
        meta_id=meta.meta_id,
        variant=variant,
        outcome=outcome,
        judgment=judgment,
        weakening_description=meta.weakening_description,
        original_statement_sha256=meta.original_statement_sha256,
        constructivized_statement_sha256=meta.constructivized_statement_sha256,
        report_digest=content_sha(payload),
    )


__all__ = [
    "CONSTRUCTIVIZATION_META_SCHEMA",
    "CONSTRUCTIVIZATION_REPORT_SCHEMA",
    "CONSTRUCTIVIZATION_VARIANT_SCHEMA",
    "ConstructivizationMetaV1",
    "ConstructivizationReportV1",
    "ConstructivizationVariantV1",
    "assert_meta_matches_task",
    "assert_not_masquerading",
    "evaluate_constructivization",
    "generate_constructivization_variant",
    "materialize_constructivized_task",
    "parse_constructivization_meta",
]
