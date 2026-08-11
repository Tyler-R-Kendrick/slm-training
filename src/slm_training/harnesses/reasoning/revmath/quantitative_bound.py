"""Quantitative-bound extraction + validation (HARN-07 / SLM-547).

Proof-mining style: turn a *qualitative* finite termination/convergence claim
into an explicit machine-checkable bound **only** when theorem assumptions
support extraction via a registered EVID-04 bound AST. Never invent a rate;
never promote empirical wall-clock timing to a theorem-derived bound.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from slm_training.formal.bound_ast import (
    BOUND_CLOSURE_LIVE_UPPER,
    BOUND_FINITE_SEARCH_PREFIX_TREE,
    BoundAstError,
    assert_registered_bound_ast_id,
    evaluate_bound,
    resolve_bound_ast,
)
from slm_training.formal.finite_search_bounds import (
    COST_MODEL_NAME,
    UNIT_COST_MODEL,
    build_finite_search_bound,
    cross_check_enumeration,
)
from slm_training.harness_core.lineage.records import content_sha
from slm_training.harnesses.reasoning.revmath.schemas import (
    RevmathSchemaError,
    RevmathTaskV1,
)

QUANT_META_SCHEMA = "revmath_quantitative_bound_meta/v1"
QUANT_REPORT_SCHEMA = "revmath_quantitative_bound_report/v1"

HermeticQuantScenario = Literal[
    "extractable",
    "nonextractable",
    "unknown",
    "malformed",
]
QuantOutcome = Literal[
    "extracted",
    "nonextractable",
    "unknown",
    "invalid",
]
KernTarget = Literal[
    "finite_search",
    "closure_live_upper",
    "none",
]


def _sorted_ids(ids: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted(dict.fromkeys(str(x) for x in ids if str(x))))


def _require_statement_digest(statement: str, digest: str, *, label: str) -> None:
    actual = content_sha(statement)
    if actual != digest:
        raise RevmathSchemaError(
            f"{label} statement_sha256 mismatch "
            f"(expected {digest!r}, got {actual!r})"
        )


@dataclass(frozen=True)
class MachineModelV1:
    """Named machine/query cost model — never a wall-clock claim."""

    name: str
    wall_clock: bool = False
    cost_per_expansion: int | None = None
    cost_per_verification: int | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if self.wall_clock:
            raise RevmathSchemaError(
                "machine_model.wall_clock must be False; "
                "theorem-derived bounds are not empirical timings"
            )
        if not self.name:
            raise RevmathSchemaError("machine_model.name must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "wall_clock": False,
            "cost_per_expansion": self.cost_per_expansion,
            "cost_per_verification": self.cost_per_verification,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MachineModelV1:
        return cls(
            name=str(data["name"]),
            wall_clock=bool(data.get("wall_clock", False)),
            cost_per_expansion=(
                int(data["cost_per_expansion"])
                if data.get("cost_per_expansion") is not None
                else None
            ),
            cost_per_verification=(
                int(data["cost_per_verification"])
                if data.get("cost_per_verification") is not None
                else None
            ),
            notes=str(data.get("notes") or ""),
        )


@dataclass(frozen=True)
class QuantitativeBoundMetaV1:
    """Frozen sidecar linking qualitative theorem → candidate bound AST."""

    schema_version: str
    meta_id: str
    qualitative_theorem_id: str
    qualitative_statement: str
    qualitative_statement_sha256: str
    theorem_ref: str
    bound_ast_id: str | None
    size_parameters: Mapping[str, Any]
    machine_model: MachineModelV1
    required_assumptions: tuple[str, ...]
    present_assumptions: tuple[str, ...]
    hermetic_scenario: HermeticQuantScenario
    kern_target: KernTarget
    lean_module: str | None = None
    lean_def: str | None = None
    expected_bound_value: str | None = None
    exhaustive_small_instance: bool = False
    domain_values_for_exhaustive: tuple[tuple[int, ...], ...] | None = None
    # Explicit notes when KERN-05 full stabilization is absent.
    missing_assumption_notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "meta_id": self.meta_id,
            "qualitative_theorem_id": self.qualitative_theorem_id,
            "qualitative_statement": self.qualitative_statement,
            "qualitative_statement_sha256": self.qualitative_statement_sha256,
            "theorem_ref": self.theorem_ref,
            "bound_ast_id": self.bound_ast_id,
            "size_parameters": dict(self.size_parameters),
            "machine_model": self.machine_model.to_dict(),
            "required_assumptions": list(self.required_assumptions),
            "present_assumptions": list(self.present_assumptions),
            "hermetic_scenario": self.hermetic_scenario,
            "kern_target": self.kern_target,
            "lean_module": self.lean_module,
            "lean_def": self.lean_def,
            "expected_bound_value": self.expected_bound_value,
            "exhaustive_small_instance": self.exhaustive_small_instance,
            "domain_values_for_exhaustive": (
                [list(row) for row in self.domain_values_for_exhaustive]
                if self.domain_values_for_exhaustive is not None
                else None
            ),
            "missing_assumption_notes": self.missing_assumption_notes,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> QuantitativeBoundMetaV1:
        if data.get("schema_version") != QUANT_META_SCHEMA:
            raise RevmathSchemaError(
                f"expected schema {QUANT_META_SCHEMA!r}, "
                f"got {data.get('schema_version')!r}"
            )
        statement = str(data["qualitative_statement"])
        digest = str(data["qualitative_statement_sha256"])
        _require_statement_digest(statement, digest, label="qualitative_statement")
        scenario = str(data.get("hermetic_scenario", ""))
        allowed = ("extractable", "nonextractable", "unknown", "malformed")
        if scenario not in allowed:
            raise RevmathSchemaError(f"unknown hermetic_scenario {scenario!r}")
        kern = str(data.get("kern_target", "none"))
        if kern not in ("finite_search", "closure_live_upper", "none"):
            raise RevmathSchemaError(f"unknown kern_target {kern!r}")
        size_raw = data.get("size_parameters") or {}
        if not isinstance(size_raw, Mapping):
            raise RevmathSchemaError("size_parameters must be an object")
        exhaustive_domains = data.get("domain_values_for_exhaustive")
        domains: tuple[tuple[int, ...], ...] | None = None
        if exhaustive_domains is not None:
            if not isinstance(exhaustive_domains, Sequence):
                raise RevmathSchemaError(
                    "domain_values_for_exhaustive must be a sequence of sequences"
                )
            domains = tuple(
                tuple(int(x) for x in row) for row in exhaustive_domains
            )
        bound_id = data.get("bound_ast_id")
        bound_ast_id = str(bound_id) if bound_id else None
        return cls(
            schema_version=QUANT_META_SCHEMA,
            meta_id=str(data["meta_id"]),
            qualitative_theorem_id=str(data["qualitative_theorem_id"]),
            qualitative_statement=statement,
            qualitative_statement_sha256=digest,
            theorem_ref=str(data["theorem_ref"]),
            bound_ast_id=bound_ast_id,
            size_parameters=dict(size_raw),
            machine_model=MachineModelV1.from_dict(data["machine_model"]),
            required_assumptions=_sorted_ids(data.get("required_assumptions") or ()),
            present_assumptions=_sorted_ids(data.get("present_assumptions") or ()),
            hermetic_scenario=scenario,  # type: ignore[arg-type]
            kern_target=kern,  # type: ignore[arg-type]
            lean_module=(
                str(data["lean_module"]) if data.get("lean_module") else None
            ),
            lean_def=str(data["lean_def"]) if data.get("lean_def") else None,
            expected_bound_value=(
                str(data["expected_bound_value"])
                if data.get("expected_bound_value") is not None
                else None
            ),
            exhaustive_small_instance=bool(
                data.get("exhaustive_small_instance", False)
            ),
            domain_values_for_exhaustive=domains,
            missing_assumption_notes=str(
                data.get("missing_assumption_notes") or ""
            ),
        )


def parse_quantitative_bound_meta(data: Mapping[str, Any]) -> QuantitativeBoundMetaV1:
    return QuantitativeBoundMetaV1.from_dict(data)


@dataclass(frozen=True)
class QuantitativeBoundReportV1:
    """Extraction report — theorem-derived vs empirical timing kept distinct."""

    schema_version: str
    meta_id: str
    task_id: str
    outcome: QuantOutcome
    qualitative_theorem_id: str
    theorem_ref: str
    bound_ast_id: str | None
    bound_value: str | None
    theorem_derived_bound: bool
    empirical_timing_claimed: bool
    missing_assumptions: tuple[str, ...]
    validated_via_bound_ast: bool
    exhaustive_small_instance_ok: bool | None
    machine_model: MachineModelV1
    lean_module: str | None
    lean_def: str | None
    kern_target: KernTarget
    detail: str = ""
    certificate_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "meta_id": self.meta_id,
            "task_id": self.task_id,
            "outcome": self.outcome,
            "qualitative_theorem_id": self.qualitative_theorem_id,
            "theorem_ref": self.theorem_ref,
            "bound_ast_id": self.bound_ast_id,
            "bound_value": self.bound_value,
            "theorem_derived_bound": self.theorem_derived_bound,
            "empirical_timing_claimed": self.empirical_timing_claimed,
            "missing_assumptions": list(self.missing_assumptions),
            "validated_via_bound_ast": self.validated_via_bound_ast,
            "exhaustive_small_instance_ok": self.exhaustive_small_instance_ok,
            "machine_model": self.machine_model.to_dict(),
            "lean_module": self.lean_module,
            "lean_def": self.lean_def,
            "kern_target": self.kern_target,
            "detail": self.detail,
            "certificate_sha256": self.certificate_sha256,
        }

    def report_digest(self) -> str:
        return content_sha(self.to_dict())


def missing_assumptions_for(
    required: Sequence[str], present: Sequence[str]
) -> tuple[str, ...]:
    present_set = set(present)
    return _sorted_ids([a for a in required if a not in present_set])


def _finite_search_cross_check(
    meta: QuantitativeBoundMetaV1, bound_value: str
) -> tuple[bool, str]:
    """Cross-check bound AST value against KERN-03 Python evidence."""

    sizes = meta.size_parameters.get("domain_sizes")
    if not isinstance(sizes, Sequence):
        return False, "size_parameters.domain_sizes missing for finite_search"
    c_exp = int(meta.size_parameters.get("c_exp", 1))
    c_ver = int(meta.size_parameters.get("c_ver", 1))
    from slm_training.formal.finite_search_bounds import ExpansionVerificationQueryModel

    model = ExpansionVerificationQueryModel(
        cost_per_expansion=c_exp, cost_per_verification=c_ver
    )
    evidence = build_finite_search_bound(sizes, cost_model=model)
    predicted = str(evidence.total_query_bound)
    if predicted != bound_value:
        return False, (
            f"KERN-03 total_query_bound {predicted} != bound AST value {bound_value}"
        )
    if meta.exhaustive_small_instance:
        domains = meta.domain_values_for_exhaustive
        if domains is None:
            return False, "exhaustive_small_instance requested without domain values"
        check = cross_check_enumeration(domains)
        if not check.get("assignments_match") or not check.get("prefix_nodes_match"):
            return False, f"exhaustive enumeration mismatch: {check}"
        if list(check["domain_sizes"]) != [len(row) for row in domains]:
            return False, "exhaustive domain size drift"
    return True, "finite_search cross-check ok"


def extract_quantitative_bound(
    task: RevmathTaskV1,
    meta: QuantitativeBoundMetaV1,
) -> QuantitativeBoundReportV1:
    """Extract + validate a quantitative bound, or refuse with missing assumptions.

    Fail closed: unknown/nonextractable never invent a rate; empirical timing
    is never reported as theorem-derived.
    """

    if task.task_kind != "quantitative_bound_extraction":
        raise RevmathSchemaError(
            f"task_kind must be quantitative_bound_extraction, got {task.task_kind!r}"
        )
    if task.proposition.statement_sha256 != meta.qualitative_statement_sha256:
        raise RevmathSchemaError(
            "task proposition digest must match qualitative_statement_sha256"
        )

    base_kwargs: dict[str, Any] = {
        "schema_version": QUANT_REPORT_SCHEMA,
        "meta_id": meta.meta_id,
        "task_id": task.task_id,
        "qualitative_theorem_id": meta.qualitative_theorem_id,
        "theorem_ref": meta.theorem_ref,
        "machine_model": meta.machine_model,
        "lean_module": meta.lean_module,
        "lean_def": meta.lean_def,
        "kern_target": meta.kern_target,
        "empirical_timing_claimed": False,
    }

    if meta.hermetic_scenario == "unknown":
        return QuantitativeBoundReportV1(
            outcome="unknown",
            bound_ast_id=meta.bound_ast_id,
            bound_value=None,
            theorem_derived_bound=False,
            missing_assumptions=(),
            validated_via_bound_ast=False,
            exhaustive_small_instance_ok=None,
            detail="extraction incomplete / timeout; no fabricated bound",
            certificate_sha256=None,
            **base_kwargs,
        )

    if meta.hermetic_scenario == "malformed":
        return QuantitativeBoundReportV1(
            outcome="invalid",
            bound_ast_id=meta.bound_ast_id,
            bound_value=None,
            theorem_derived_bound=False,
            missing_assumptions=(),
            validated_via_bound_ast=False,
            exhaustive_small_instance_ok=None,
            detail="malformed extraction payload",
            certificate_sha256=None,
            **base_kwargs,
        )

    missing = missing_assumptions_for(
        meta.required_assumptions, meta.present_assumptions
    )
    if missing or meta.hermetic_scenario == "nonextractable":
        notes = meta.missing_assumption_notes
        detail = (
            "nonextractable: missing assumptions "
            + ",".join(missing if missing else ("(scenario)",))
        )
        if notes:
            detail = f"{detail}; {notes}"
        return QuantitativeBoundReportV1(
            outcome="nonextractable",
            bound_ast_id=None,  # never report a fabricated bound id as extracted
            bound_value=None,
            theorem_derived_bound=False,
            missing_assumptions=missing,
            validated_via_bound_ast=False,
            exhaustive_small_instance_ok=None,
            detail=detail,
            certificate_sha256=None,
            **base_kwargs,
        )

    if not meta.bound_ast_id:
        return QuantitativeBoundReportV1(
            outcome="nonextractable",
            bound_ast_id=None,
            bound_value=None,
            theorem_derived_bound=False,
            missing_assumptions=missing_assumptions_for(
                meta.required_assumptions, meta.present_assumptions
            ),
            validated_via_bound_ast=False,
            exhaustive_small_instance_ok=None,
            detail="no candidate bound_ast_id; refusing to invent a rate",
            certificate_sha256=None,
            **base_kwargs,
        )

    try:
        assert_registered_bound_ast_id(meta.bound_ast_id)
        doc = resolve_bound_ast(meta.bound_ast_id)
        if not doc.evaluable:
            return QuantitativeBoundReportV1(
                outcome="nonextractable",
                bound_ast_id=meta.bound_ast_id,
                bound_value=None,
                theorem_derived_bound=False,
                missing_assumptions=(),
                validated_via_bound_ast=False,
                exhaustive_small_instance_ok=None,
                detail=(
                    f"bound_ast_id {meta.bound_ast_id!r} registered but not evaluable"
                ),
                certificate_sha256=None,
                **base_kwargs,
            )
        if doc.wall_clock_claim:
            return QuantitativeBoundReportV1(
                outcome="invalid",
                bound_ast_id=meta.bound_ast_id,
                bound_value=None,
                theorem_derived_bound=False,
                missing_assumptions=(),
                validated_via_bound_ast=False,
                exhaustive_small_instance_ok=None,
                detail="bound AST marks wall_clock_claim; refuse theorem-derived report",
                certificate_sha256=None,
                **base_kwargs,
            )
        result = evaluate_bound(meta.bound_ast_id, meta.size_parameters)
    except BoundAstError as exc:
        return QuantitativeBoundReportV1(
            outcome="invalid",
            bound_ast_id=meta.bound_ast_id,
            bound_value=None,
            theorem_derived_bound=False,
            missing_assumptions=(),
            validated_via_bound_ast=False,
            exhaustive_small_instance_ok=None,
            detail=f"bound AST validation failed: {exc}",
            certificate_sha256=None,
            **base_kwargs,
        )

    bound_value = result.value
    if (
        meta.expected_bound_value is not None
        and bound_value != meta.expected_bound_value
    ):
        return QuantitativeBoundReportV1(
            outcome="invalid",
            bound_ast_id=meta.bound_ast_id,
            bound_value=bound_value,
            theorem_derived_bound=False,
            missing_assumptions=(),
            validated_via_bound_ast=True,
            exhaustive_small_instance_ok=None,
            detail=(
                f"bound value {bound_value!r} != fixture expected "
                f"{meta.expected_bound_value!r}"
            ),
            certificate_sha256=None,
            **base_kwargs,
        )

    exhaustive_ok: bool | None = None
    detail = f"extracted via {meta.bound_ast_id}"
    if meta.kern_target == "finite_search":
        if meta.bound_ast_id != BOUND_FINITE_SEARCH_PREFIX_TREE:
            return QuantitativeBoundReportV1(
                outcome="invalid",
                bound_ast_id=meta.bound_ast_id,
                bound_value=bound_value,
                theorem_derived_bound=False,
                missing_assumptions=(),
                validated_via_bound_ast=True,
                exhaustive_small_instance_ok=None,
                detail=(
                    "finite_search kern_target requires "
                    f"{BOUND_FINITE_SEARCH_PREFIX_TREE}"
                ),
                certificate_sha256=None,
                **base_kwargs,
            )
        ok, msg = _finite_search_cross_check(meta, bound_value)
        if not ok:
            return QuantitativeBoundReportV1(
                outcome="invalid",
                bound_ast_id=meta.bound_ast_id,
                bound_value=bound_value,
                theorem_derived_bound=False,
                missing_assumptions=(),
                validated_via_bound_ast=True,
                exhaustive_small_instance_ok=False,
                detail=msg,
                certificate_sha256=None,
                **base_kwargs,
            )
        exhaustive_ok = True if meta.exhaustive_small_instance else None
        detail = msg
    elif meta.kern_target == "closure_live_upper":
        if meta.bound_ast_id != BOUND_CLOSURE_LIVE_UPPER:
            return QuantitativeBoundReportV1(
                outcome="invalid",
                bound_ast_id=meta.bound_ast_id,
                bound_value=bound_value,
                theorem_derived_bound=False,
                missing_assumptions=(),
                validated_via_bound_ast=True,
                exhaustive_small_instance_ok=None,
                detail=(
                    "closure_live_upper kern_target requires "
                    f"{BOUND_CLOSURE_LIVE_UPPER}"
                ),
                certificate_sha256=None,
                **base_kwargs,
            )
        # Full KERN-05 stabilization proofs may still be backlog; live_upper is
        # the registered ExactClosure survivor cardinality bound.
        detail = (
            "closure live_upper via ExactClosure.closePass_subset; "
            "not a wall-clock claim; full KERN-05 stabilization may still be open"
        )

    cert_payload = {
        "bound_ast_id": meta.bound_ast_id,
        "bound_value": bound_value,
        "size_parameters": dict(meta.size_parameters),
        "theorem_ref": meta.theorem_ref,
        "machine_model": meta.machine_model.to_dict(),
        "u64_overflow": result.u64_overflow,
    }
    return QuantitativeBoundReportV1(
        outcome="extracted",
        bound_ast_id=meta.bound_ast_id,
        bound_value=bound_value,
        theorem_derived_bound=True,
        missing_assumptions=(),
        validated_via_bound_ast=True,
        exhaustive_small_instance_ok=exhaustive_ok,
        detail=detail,
        certificate_sha256=content_sha(cert_payload),
        **base_kwargs,
    )


def default_finite_search_machine_model() -> MachineModelV1:
    return MachineModelV1(
        name=COST_MODEL_NAME,
        wall_clock=False,
        cost_per_expansion=UNIT_COST_MODEL.cost_per_expansion,
        cost_per_verification=UNIT_COST_MODEL.cost_per_verification,
        notes="KERN-03 ExpansionVerificationQueryModel; query counts only",
    )


def default_closure_machine_model() -> MachineModelV1:
    return MachineModelV1(
        name="ExactClosureLiveCardinality",
        wall_clock=False,
        notes="Survivor count ≤ pre-closure live cardinality (closePass_subset)",
    )


__all__ = [
    "QUANT_META_SCHEMA",
    "QUANT_REPORT_SCHEMA",
    "MachineModelV1",
    "QuantitativeBoundMetaV1",
    "QuantitativeBoundReportV1",
    "default_closure_machine_model",
    "default_finite_search_machine_model",
    "extract_quantitative_bound",
    "missing_assumptions_for",
    "parse_quantitative_bound_meta",
]
