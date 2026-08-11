"""INTEG-04 / SLM-529: opt-in four-axis ledger projection for harness reports.

Projects EVID-03 ``RevmathFourAxisAnalysisV1`` / ``FormalPreflightFourAxisLedgerV1``,
KERN-12 computability labels, and INTEG-01 / KERN-11 refinement refs into existing
harness report/evidence envelopes. Default-off: callers that leave ``enabled=False``
keep byte-identical report payloads (no projection key). Absent formal evidence is
recorded as ``absent`` / ``unknown`` — never as failure or proof.

Does **not** introduce a parallel revmath reporting stack, campaign store, or
solver path. Harnesses must not claim a stronger ``authority_class`` than the
underlying formal object/checker supplies.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from slm_training.formal.authority import AuthorityClassName
from slm_training.harness_core.lineage.records import content_sha

SCHEMA = "harness_four_axis_ledger_projection/v1"
ISSUE = "SLM-529"
KEY = "INTEG-04"
PROJECTION_FIELD = "four_axis_ledger_projection"

# Families covered by parity fixtures (reasoning + non-reasoning).
HarnessLedgerFamily = Literal[
    "reasoning/revmath",
    "train_data",
    "model_build",
]

AxisPresence = Literal["absent", "unknown", "referenced"]


_AUTHORITY_RANK: dict[str, int] = {
    "invalid": 0,
    "unchecked": 1,
    "empirical_remainder": 2,
    "checked_semantic": 3,
}

_KNOWN_FAMILIES: frozenset[str] = frozenset(
    {"reasoning/revmath", "train_data", "model_build"}
)


class FourAxisLedgerAdapterError(ValueError):
    """Malformed projection, invented authority, or family mismatch."""


@dataclass(frozen=True)
class AxisProjectionRefV1:
    """One axis slot: presence + optional digests/refs (never invented proofs)."""

    axis: str
    presence: AxisPresence
    status: str | None = None
    certificate_sha256: str | None = None
    theorem_ref: str | None = None
    bound_ast_id: str | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "axis": self.axis,
            "presence": self.presence,
            "status": self.status,
            "certificate_sha256": self.certificate_sha256,
            "theorem_ref": self.theorem_ref,
            "bound_ast_id": self.bound_ast_id,
            "notes": self.notes,
        }
        return {k: v for k, v in payload.items() if v is not None and v != ""}


@dataclass(frozen=True)
class HarnessFourAxisLedgerProjectionV1:
    """Opt-in four-axis formal-analysis projection for one harness report."""

    schema_version: str
    issue_key: str
    linear_id: str
    harness_family: str
    logical_strength: AxisProjectionRefV1
    computability: AxisProjectionRefV1
    resource_bound: AxisProjectionRefV1
    refinement_empirical_remainder: AxisProjectionRefV1
    computability_class_id: str | None
    rm_subsystem_id: str | None
    refinement_trace_hash: str | None
    refinement_certificate_sha256: str | None
    authority_ceiling: AuthorityClassName | None
    formal_preflight_sha256: str | None
    content_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "issue_key": self.issue_key,
            "linear_id": self.linear_id,
            "harness_family": self.harness_family,
            "logical_strength": self.logical_strength.to_dict(),
            "computability": self.computability.to_dict(),
            "resource_bound": self.resource_bound.to_dict(),
            "refinement_empirical_remainder": self.refinement_empirical_remainder.to_dict(),
            "computability_class_id": self.computability_class_id,
            "rm_subsystem_id": self.rm_subsystem_id,
            "refinement_trace_hash": self.refinement_trace_hash,
            "refinement_certificate_sha256": self.refinement_certificate_sha256,
            "authority_ceiling": self.authority_ceiling,
            "formal_preflight_sha256": self.formal_preflight_sha256,
            "content_sha256": self.content_sha256,
        }


def _axis_from_mapping(name: str, raw: Mapping[str, Any] | None) -> AxisProjectionRefV1:
    if raw is None:
        return AxisProjectionRefV1(axis=name, presence="absent")
    status = raw.get("status")
    cert = (
        None
        if raw.get("certificate_sha256") is None
        else str(raw["certificate_sha256"])
    )
    theorem = None if raw.get("theorem_ref") is None else str(raw["theorem_ref"])
    bound = None if raw.get("bound_ast_id") is None else str(raw["bound_ast_id"])
    notes = str(raw.get("notes") or "")
    if "presence" in raw:
        presence = str(raw["presence"])
        if presence not in ("absent", "unknown", "referenced"):
            raise FourAxisLedgerAdapterError(
                f"axis {name!r} has invalid presence {presence!r}"
            )
        return AxisProjectionRefV1(
            axis=name,
            presence=presence,  # type: ignore[arg-type]
            status=None if status is None else str(status),
            certificate_sha256=cert,
            theorem_ref=theorem,
            bound_ast_id=bound,
            notes=notes,
        )
    if status is None or status == "not_claimed":
        return AxisProjectionRefV1(axis=name, presence="absent", status=status)
    if status == "unknown_axis":
        return AxisProjectionRefV1(
            axis=name,
            presence="unknown",
            status=status,
            notes=notes,
        )
    # proved / refuted / empirical_remainder — reference only; never invent proof.
    return AxisProjectionRefV1(
        axis=name,
        presence="referenced",
        status=str(status),
        certificate_sha256=cert,
        theorem_ref=theorem,
        bound_ast_id=bound,
        notes=notes,
    )


def _axis_from_obj(name: str, axis: Any | None) -> AxisProjectionRefV1:
    if axis is None:
        return AxisProjectionRefV1(axis=name, presence="absent")
    if hasattr(axis, "model_dump"):
        return _axis_from_mapping(name, axis.model_dump(mode="json"))
    if isinstance(axis, Mapping):
        return _axis_from_mapping(name, axis)
    raise FourAxisLedgerAdapterError(f"unsupported axis payload for {name!r}")


def _ceiling_from_axes(
    axes: tuple[AxisProjectionRefV1, ...],
    *,
    formal_authority_class: AuthorityClassName | None,
) -> AuthorityClassName | None:
    """Derive max claimable authority; never above the formal object/checker."""

    if all(a.presence == "absent" for a in axes) and formal_authority_class is None:
        return None
    derived: AuthorityClassName = "unchecked"
    for axis in axes:
        if axis.presence == "absent":
            continue
        if axis.status == "empirical_remainder":
            if _AUTHORITY_RANK["empirical_remainder"] > _AUTHORITY_RANK[derived]:
                derived = "empirical_remainder"
        # proved/refuted/unknown stay unchecked unless formal authority upgrades
    if formal_authority_class is None:
        return derived if any(a.presence != "absent" for a in axes) else None
    if _AUTHORITY_RANK[formal_authority_class] < _AUTHORITY_RANK[derived]:
        # Formal object is weaker than axis-derived remainder → keep formal.
        return formal_authority_class
    return formal_authority_class


def _reject_stronger(
    claimed: AuthorityClassName | None,
    ceiling: AuthorityClassName | None,
) -> AuthorityClassName | None:
    if claimed is None:
        return ceiling
    if ceiling is None:
        raise FourAxisLedgerAdapterError(
            "cannot claim authority_class without underlying formal evidence"
        )
    if _AUTHORITY_RANK[claimed] > _AUTHORITY_RANK[ceiling]:
        raise FourAxisLedgerAdapterError(
            f"harness cannot invent authority_class {claimed!r} above "
            f"underlying ceiling {ceiling!r}"
        )
    return claimed


def project_four_axis_ledger(
    *,
    harness_family: str,
    analysis: Any | Mapping[str, Any] | None = None,
    ledger: Any | Mapping[str, Any] | None = None,
    computability_class_id: str | None = None,
    refinement_trace_hash: str | None = None,
    refinement_certificate_sha256: str | None = None,
    formal_authority_class: AuthorityClassName | None = None,
    claimed_authority_class: AuthorityClassName | None = None,
) -> HarnessFourAxisLedgerProjectionV1:
    """Build a projection. Missing evidence → absent/unknown slots (not failure)."""

    if harness_family not in _KNOWN_FAMILIES:
        raise FourAxisLedgerAdapterError(
            f"unknown harness_family {harness_family!r}; "
            f"expected one of {sorted(_KNOWN_FAMILIES)}"
        )

    analysis_payload: Mapping[str, Any] | None = None
    rm_subsystem_id: str | None = None
    formal_preflight_sha256: str | None = None
    class_id = computability_class_id

    if ledger is not None:
        if hasattr(ledger, "model_dump"):
            ledger_data = ledger.model_dump(mode="json")
        elif isinstance(ledger, Mapping):
            ledger_data = dict(ledger)
        else:
            raise FourAxisLedgerAdapterError("ledger must be a mapping or ledger model")
        analysis_payload = ledger_data.get("analysis")  # type: ignore[assignment]
        rm_subsystem_id = ledger_data.get("rm_subsystem_id")
        if class_id is None:
            class_id = ledger_data.get("computability_classification")
        # ledger alone does not prove checked_semantic
    if analysis is not None:
        if hasattr(analysis, "model_dump"):
            analysis_payload = analysis.model_dump(mode="json")
        elif isinstance(analysis, Mapping):
            analysis_payload = dict(analysis)
        else:
            raise FourAxisLedgerAdapterError("analysis must be a mapping or four-axis model")

    if analysis_payload is None:
        logical = AxisProjectionRefV1(axis="assumption_strength", presence="absent")
        computability = AxisProjectionRefV1(axis="computability", presence="absent")
        resource = AxisProjectionRefV1(axis="resource_bounds", presence="absent")
        refinement = AxisProjectionRefV1(
            axis="implementation_refinement", presence="absent"
        )
    else:
        if analysis_payload.get("schema_version") not in (None, "revmath_four_axis/v1"):
            raise FourAxisLedgerAdapterError(
                f"unsupported four-axis schema {analysis_payload.get('schema_version')!r}"
            )
        logical = _axis_from_mapping(
            "assumption_strength", analysis_payload.get("assumption_strength")  # type: ignore[arg-type]
        )
        computability = _axis_from_mapping(
            "computability", analysis_payload.get("computability")  # type: ignore[arg-type]
        )
        resource = _axis_from_mapping(
            "resource_bounds", analysis_payload.get("resource_bounds")  # type: ignore[arg-type]
        )
        refinement = _axis_from_mapping(
            "implementation_refinement",
            analysis_payload.get("implementation_refinement"),  # type: ignore[arg-type]
        )
        formal_preflight_sha256 = (
            None
            if analysis_payload.get("formal_preflight_sha256") is None
            else str(analysis_payload["formal_preflight_sha256"])
        )

    # Optional KERN-12 / INTEG-01/11 refs — attach without upgrading axis presence.
    if class_id is not None and computability.presence == "absent":
        computability = AxisProjectionRefV1(
            axis="computability",
            presence="referenced",
            status="not_claimed",
            notes="computability_class_id reference only (KERN-12)",
        )
    if (
        refinement_trace_hash is not None or refinement_certificate_sha256 is not None
    ) and refinement.presence == "absent":
        refinement = AxisProjectionRefV1(
            axis="implementation_refinement",
            presence="referenced",
            status="empirical_remainder"
            if refinement_certificate_sha256 is None
            else "not_claimed",
            notes="INTEG-01/KERN-11 refinement reference only",
        )

    axes = (logical, computability, resource, refinement)
    ceiling = _ceiling_from_axes(axes, formal_authority_class=formal_authority_class)
    ceiling = _reject_stronger(claimed_authority_class, ceiling)

    projection = HarnessFourAxisLedgerProjectionV1(
        schema_version=SCHEMA,
        issue_key=KEY,
        linear_id=ISSUE,
        harness_family=harness_family,
        logical_strength=logical,
        computability=computability,
        resource_bound=resource,
        refinement_empirical_remainder=refinement,
        computability_class_id=class_id,
        rm_subsystem_id=rm_subsystem_id,
        refinement_trace_hash=refinement_trace_hash,
        refinement_certificate_sha256=refinement_certificate_sha256,
        authority_ceiling=ceiling,
        formal_preflight_sha256=formal_preflight_sha256,
        content_sha256="",  # filled below
    )
    unsigned = {k: v for k, v in projection.to_dict().items() if k != "content_sha256"}
    digest = content_sha(unsigned)
    return HarnessFourAxisLedgerProjectionV1(
        schema_version=projection.schema_version,
        issue_key=projection.issue_key,
        linear_id=projection.linear_id,
        harness_family=projection.harness_family,
        logical_strength=projection.logical_strength,
        computability=projection.computability,
        resource_bound=projection.resource_bound,
        refinement_empirical_remainder=projection.refinement_empirical_remainder,
        computability_class_id=projection.computability_class_id,
        rm_subsystem_id=projection.rm_subsystem_id,
        refinement_trace_hash=projection.refinement_trace_hash,
        refinement_certificate_sha256=projection.refinement_certificate_sha256,
        authority_ceiling=projection.authority_ceiling,
        formal_preflight_sha256=projection.formal_preflight_sha256,
        content_sha256=digest,
    )


def validate_harness_four_axis_projection(
    payload: Mapping[str, Any],
) -> HarnessFourAxisLedgerProjectionV1:
    """Fail closed on malformed or digest-mismatched projections."""

    if not isinstance(payload, Mapping):
        raise FourAxisLedgerAdapterError("projection must be a JSON object")
    if payload.get("schema_version") != SCHEMA:
        raise FourAxisLedgerAdapterError(
            f"unsupported projection schema {payload.get('schema_version')!r}"
        )
    family = str(payload.get("harness_family") or "")
    if family not in _KNOWN_FAMILIES:
        raise FourAxisLedgerAdapterError(f"unknown harness_family {family!r}")
    for key in (
        "logical_strength",
        "computability",
        "resource_bound",
        "refinement_empirical_remainder",
    ):
        if key not in payload or not isinstance(payload[key], Mapping):
            raise FourAxisLedgerAdapterError(f"projection missing axis slot {key!r}")
    unsigned = {k: v for k, v in payload.items() if k != "content_sha256"}
    expected = content_sha(unsigned)
    digest = payload.get("content_sha256")
    if digest is None or str(digest) != expected:
        raise FourAxisLedgerAdapterError("projection content_sha256 mismatch")
    ceiling = payload.get("authority_ceiling")
    if ceiling is not None and ceiling not in _AUTHORITY_RANK:
        raise FourAxisLedgerAdapterError(f"unknown authority_ceiling {ceiling!r}")
    return HarnessFourAxisLedgerProjectionV1(
        schema_version=SCHEMA,
        issue_key=str(payload.get("issue_key") or KEY),
        linear_id=str(payload.get("linear_id") or ISSUE),
        harness_family=family,
        logical_strength=_axis_from_mapping(
            "assumption_strength", payload["logical_strength"]  # type: ignore[arg-type]
        ),
        computability=_axis_from_mapping(
            "computability", payload["computability"]  # type: ignore[arg-type]
        ),
        resource_bound=_axis_from_mapping(
            "resource_bounds", payload["resource_bound"]  # type: ignore[arg-type]
        ),
        refinement_empirical_remainder=_axis_from_mapping(
            "implementation_refinement",
            payload["refinement_empirical_remainder"],  # type: ignore[arg-type]
        ),
        computability_class_id=(
            None
            if payload.get("computability_class_id") is None
            else str(payload["computability_class_id"])
        ),
        rm_subsystem_id=(
            None if payload.get("rm_subsystem_id") is None else str(payload["rm_subsystem_id"])
        ),
        refinement_trace_hash=(
            None
            if payload.get("refinement_trace_hash") is None
            else str(payload["refinement_trace_hash"])
        ),
        refinement_certificate_sha256=(
            None
            if payload.get("refinement_certificate_sha256") is None
            else str(payload["refinement_certificate_sha256"])
        ),
        authority_ceiling=ceiling,  # type: ignore[arg-type]
        formal_preflight_sha256=(
            None
            if payload.get("formal_preflight_sha256") is None
            else str(payload["formal_preflight_sha256"])
        ),
        content_sha256=str(digest),
    )


def attach_four_axis_ledger_projection(
    report: Mapping[str, Any],
    *,
    enabled: bool = False,
    harness_family: str,
    analysis: Any | Mapping[str, Any] | None = None,
    ledger: Any | Mapping[str, Any] | None = None,
    computability_class_id: str | None = None,
    refinement_trace_hash: str | None = None,
    refinement_certificate_sha256: str | None = None,
    formal_authority_class: AuthorityClassName | None = None,
    claimed_authority_class: AuthorityClassName | None = None,
    field_name: str = PROJECTION_FIELD,
) -> dict[str, Any]:
    """Attach or omit the projection. ``enabled=False`` leaves the report unchanged.

    Pure report-seam adapter: does not change model/solver behavior.
    """

    out = dict(report)
    if not enabled:
        out.pop(field_name, None)
        return out
    projection = project_four_axis_ledger(
        harness_family=harness_family,
        analysis=analysis,
        ledger=ledger,
        computability_class_id=computability_class_id,
        refinement_trace_hash=refinement_trace_hash,
        refinement_certificate_sha256=refinement_certificate_sha256,
        formal_authority_class=formal_authority_class,
        claimed_authority_class=claimed_authority_class,
    )
    out[field_name] = projection.to_dict()
    return out


def attach_to_evaluation_report_metadata(
    metadata: Mapping[str, Any] | None,
    *,
    enabled: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """model_build ``EvaluationReport.metadata`` seam (non-reasoning)."""

    base = dict(metadata or {})
    return attach_four_axis_ledger_projection(
        base,
        enabled=enabled,
        harness_family="model_build",
        **kwargs,
    )


__all__ = [
    "SCHEMA",
    "ISSUE",
    "KEY",
    "PROJECTION_FIELD",
    "AxisPresence",
    "AxisProjectionRefV1",
    "FourAxisLedgerAdapterError",
    "HarnessFourAxisLedgerProjectionV1",
    "HarnessLedgerFamily",
    "attach_four_axis_ledger_projection",
    "attach_to_evaluation_report_metadata",
    "project_four_axis_ledger",
    "validate_harness_four_axis_projection",
]
