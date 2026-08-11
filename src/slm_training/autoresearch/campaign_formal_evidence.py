"""INTEG-03: link four-axis / revmath / FormalAuthorityV2 into campaign results.

Optional, content-bound refs on existing ``CampaignResultV1`` / dispositions.
Formal and empirical statuses persist independently — a proof never authorizes
ship or promotion success by itself. Historical results without the split
field remain valid (migration leaves ``formal_empirical=None``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import Field, model_validator

from slm_training.autoresearch.schemas import StrictModel
from slm_training.lineage.records import content_sha

FORMAL_EMPIRICAL_SPLIT_SCHEMA = "campaign_formal_empirical_split/v1"
DECISION_SUPPORT_SCHEMA = "campaign_decision_support/v1"
CONTENT_BOUND_REF_SCHEMA = "campaign_content_bound_ref/v1"

FormalEvidenceStatus = Literal[
    "proved",
    "refuted",
    "unknown",
    "weak",
    "unchecked",
    "invalid",
    "empirical_remainder",
    "absent",
    "not_applicable",
]

EmpiricalEvidenceStatus = Literal[
    "success",
    "failure",
    "inconclusive",
    "not_run",
    "exploratory",
]

EvidenceRefKind = Literal[
    "formal_authority_v2",
    "formal_preflight",
    "four_axis_ledger",
    "revmath_report",
    "revmath_result",
    "bound_ast",
    "runtime_refinement_trace",
    "empirical_remainder_claim",
    "canonical_proof_trace",
]

_HEX64 = r"^[0-9a-f]{64}$"

# Formal outcomes that are "strong" — still cannot set empirical success.
_STRONG_FORMAL: frozenset[str] = frozenset({"proved", "refuted"})
_WEAK_FORMAL: frozenset[str] = frozenset(
    {"unknown", "weak", "unchecked", "absent", "empirical_remainder", "not_applicable"}
)


class ContentBoundEvidenceRefV1(StrictModel):
    """One content-addressed evidence pointer (replayable; no parallel store)."""

    schema_version: Literal["campaign_content_bound_ref/v1"] = CONTENT_BOUND_REF_SCHEMA
    kind: EvidenceRefKind
    sha256: str = Field(pattern=_HEX64)
    uri: str | None = Field(default=None, min_length=1)
    label: str | None = Field(default=None, min_length=1)
    notes: str = ""


class CampaignDecisionSupportV1(StrictModel):
    """Which assumptions / bounds / refinement evidence support a decision."""

    schema_version: Literal["campaign_decision_support/v1"] = DECISION_SUPPORT_SCHEMA
    assumption_axis_status: str | None = None
    computability_axis_status: str | None = None
    resource_bounds_axis_status: str | None = None
    refinement_axis_status: str | None = None
    bound_ast_ids: tuple[str, ...] = ()
    formal_authority_digests: tuple[str, ...] = ()
    four_axis_ledger_sha256: str | None = Field(default=None, pattern=_HEX64)
    revmath_report_sha256: str | None = Field(default=None, pattern=_HEX64)
    runtime_refinement_trace_sha256: str | None = Field(default=None, pattern=_HEX64)
    empirical_remainder_claim_ids: tuple[str, ...] = ()
    notes: str = ""


class CampaignFormalEmpiricalSplitV1(StrictModel):
    """Independent formal vs empirical statuses with content-bound refs.

    ``content_digest`` covers every field except itself. Promotion / ship
    eligibility is never derived from ``formal_status`` alone.
    """

    schema_version: Literal["campaign_formal_empirical_split/v1"] = (
        FORMAL_EMPIRICAL_SPLIT_SCHEMA
    )
    formal_status: FormalEvidenceStatus
    empirical_status: EmpiricalEvidenceStatus
    refs: tuple[ContentBoundEvidenceRefV1, ...] = ()
    empirical_remainder_claim_ids: tuple[str, ...] = ()
    decision_support: CampaignDecisionSupportV1 | None = None
    content_digest: str = Field(pattern=_HEX64)

    @model_validator(mode="after")
    def _digest_and_invariants(self) -> CampaignFormalEmpiricalSplitV1:
        recomputed = _split_content_digest(self)
        if recomputed != self.content_digest:
            raise ValueError(
                "formal/empirical split content_digest mismatch "
                f"(declared={self.content_digest}, recomputed={recomputed})"
            )
        if (
            self.formal_status == "empirical_remainder"
            and not self.empirical_remainder_claim_ids
            and not (
                self.decision_support
                and self.decision_support.empirical_remainder_claim_ids
            )
        ):
            raise ValueError(
                "empirical_remainder formal_status requires empirical_remainder_claim_ids"
            )
        ref_kinds = {ref.kind for ref in self.refs}
        if self.decision_support is not None:
            ds = self.decision_support
            if ds.four_axis_ledger_sha256 and "four_axis_ledger" not in ref_kinds:
                raise ValueError(
                    "decision_support.four_axis_ledger_sha256 requires a four_axis_ledger ref"
                )
            if ds.revmath_report_sha256 and "revmath_report" not in ref_kinds:
                raise ValueError(
                    "decision_support.revmath_report_sha256 requires a revmath_report ref"
                )
            if (
                ds.runtime_refinement_trace_sha256
                and "runtime_refinement_trace" not in ref_kinds
            ):
                raise ValueError(
                    "decision_support.runtime_refinement_trace_sha256 requires a "
                    "runtime_refinement_trace ref"
                )
        return self


def _split_body(split: CampaignFormalEmpiricalSplitV1 | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(split, CampaignFormalEmpiricalSplitV1):
        payload = split.model_dump(mode="json")
    else:
        payload = dict(split)
    payload.pop("content_digest", None)
    return payload


def _split_content_digest(
    split: CampaignFormalEmpiricalSplitV1 | Mapping[str, Any],
) -> str:
    return content_sha(_split_body(split))


def build_formal_empirical_split(
    *,
    formal_status: FormalEvidenceStatus,
    empirical_status: EmpiricalEvidenceStatus,
    refs: Sequence[ContentBoundEvidenceRefV1 | Mapping[str, Any]] = (),
    empirical_remainder_claim_ids: Sequence[str] = (),
    decision_support: CampaignDecisionSupportV1 | Mapping[str, Any] | None = None,
) -> CampaignFormalEmpiricalSplitV1:
    """Construct a sealed split (preferred write path)."""

    parsed_refs = tuple(
        item
        if isinstance(item, ContentBoundEvidenceRefV1)
        else ContentBoundEvidenceRefV1.model_validate(item)
        for item in refs
    )
    parsed_support: CampaignDecisionSupportV1 | None
    if decision_support is None:
        parsed_support = None
    elif isinstance(decision_support, CampaignDecisionSupportV1):
        parsed_support = decision_support
    else:
        parsed_support = CampaignDecisionSupportV1.model_validate(decision_support)
    provisional = {
        "schema_version": FORMAL_EMPIRICAL_SPLIT_SCHEMA,
        "formal_status": formal_status,
        "empirical_status": empirical_status,
        "refs": [ref.model_dump(mode="json") for ref in parsed_refs],
        "empirical_remainder_claim_ids": list(empirical_remainder_claim_ids),
        "decision_support": (
            None if parsed_support is None else parsed_support.model_dump(mode="json")
        ),
    }
    digest = _split_content_digest(provisional)
    return CampaignFormalEmpiricalSplitV1.model_validate(
        {**provisional, "content_digest": digest}
    )


def formal_status_from_authority(authority: Mapping[str, Any] | Any) -> FormalEvidenceStatus:
    """Map FormalAuthorityV2 (dict or object) to an independent formal status."""

    if hasattr(authority, "to_dict"):
        data = authority.to_dict()
    else:
        data = dict(authority)
    klass = str(data.get("authority_class") or "")
    if klass == "empirical_remainder":
        return "empirical_remainder"
    if klass == "invalid":
        return "invalid"
    if klass == "unchecked":
        return "unchecked"
    judgment = data.get("judgment") or {}
    if hasattr(judgment, "to_dict"):
        judgment = judgment.to_dict()
    outcome = str(judgment.get("outcome") or "").lower()
    if outcome in {"proved", "proven", "certified", "witnessed"}:
        return "proved"
    if outcome in {"refuted", "falsified"}:
        return "refuted"
    if outcome in {"unknown", "incomplete", "skipped"}:
        return "unknown"
    if outcome == "invalid":
        return "invalid"
    if klass == "checked_semantic":
        return "weak"
    return "unknown"


def empirical_status_from_result(
    *,
    ship_gates_passed: bool,
    exploratory: bool,
    claim_class: str,
    endpoint_success: bool | None = None,
    runs_failed: bool = False,
) -> EmpiricalEvidenceStatus:
    """Derive empirical status without consulting formal evidence."""

    if exploratory or claim_class in {"fixture", "wiring", "diagnostic"}:
        return "exploratory"
    if runs_failed:
        return "failure"
    if endpoint_success is False:
        return "failure"
    if ship_gates_passed or endpoint_success is True:
        return "success"
    if endpoint_success is None and not ship_gates_passed:
        return "inconclusive"
    return "not_run"


def link_four_axis_and_revmath(
    *,
    formal_status: FormalEvidenceStatus,
    empirical_status: EmpiricalEvidenceStatus,
    formal_authority_digests: Sequence[str] = (),
    formal_preflight_digests: Sequence[str] = (),
    four_axis_ledger_sha256: str | None = None,
    revmath_report_sha256: str | None = None,
    revmath_result_digests: Sequence[str] = (),
    bound_ast_ids: Sequence[str] = (),
    bound_ast_digests: Sequence[tuple[str, str]] = (),
    runtime_refinement_trace_sha256: str | None = None,
    empirical_remainder_claim_ids: Sequence[str] = (),
    assumption_axis_status: str | None = None,
    computability_axis_status: str | None = None,
    resource_bounds_axis_status: str | None = None,
    refinement_axis_status: str | None = None,
    notes: str = "",
) -> CampaignFormalEmpiricalSplitV1:
    """Build the optional campaign linkage from existing owners' digests.

    ``runtime_refinement_trace_sha256`` is an optional slot for KERN-11
    (SLM-539); omit until that digest exists — never invent a refinement claim.
    """

    refs: list[ContentBoundEvidenceRefV1] = []
    for digest in formal_authority_digests:
        refs.append(
            ContentBoundEvidenceRefV1(
                kind="formal_authority_v2",
                sha256=str(digest),
                label="formal_authority/v2",
            )
        )
    for digest in formal_preflight_digests:
        refs.append(
            ContentBoundEvidenceRefV1(
                kind="formal_preflight",
                sha256=str(digest),
                label="formal_preflight/v1",
            )
        )
    if four_axis_ledger_sha256:
        refs.append(
            ContentBoundEvidenceRefV1(
                kind="four_axis_ledger",
                sha256=four_axis_ledger_sha256,
                label="formal_preflight_four_axis_ledger/v1",
            )
        )
    if revmath_report_sha256:
        refs.append(
            ContentBoundEvidenceRefV1(
                kind="revmath_report",
                sha256=revmath_report_sha256,
                label="revmath_report/v1",
            )
        )
    for digest in revmath_result_digests:
        refs.append(
            ContentBoundEvidenceRefV1(
                kind="revmath_result",
                sha256=str(digest),
                label="revmath_result/v1",
            )
        )
    for bound_id, digest in bound_ast_digests:
        refs.append(
            ContentBoundEvidenceRefV1(
                kind="bound_ast",
                sha256=str(digest),
                label=str(bound_id),
            )
        )
    if runtime_refinement_trace_sha256:
        refs.append(
            ContentBoundEvidenceRefV1(
                kind="runtime_refinement_trace",
                sha256=runtime_refinement_trace_sha256,
                label="kern11_refinement_slot",
                notes="optional KERN-11 digest slot; verification owned elsewhere",
            )
        )
    for claim_id in empirical_remainder_claim_ids:
        # Content-bind the claim id string itself so the remainder is replayable.
        refs.append(
            ContentBoundEvidenceRefV1(
                kind="empirical_remainder_claim",
                sha256=content_sha(str(claim_id)),
                label=str(claim_id),
            )
        )

    support = CampaignDecisionSupportV1(
        assumption_axis_status=assumption_axis_status,
        computability_axis_status=computability_axis_status,
        resource_bounds_axis_status=resource_bounds_axis_status,
        refinement_axis_status=refinement_axis_status,
        bound_ast_ids=tuple(str(item) for item in bound_ast_ids),
        formal_authority_digests=tuple(str(item) for item in formal_authority_digests),
        four_axis_ledger_sha256=four_axis_ledger_sha256,
        revmath_report_sha256=revmath_report_sha256,
        runtime_refinement_trace_sha256=runtime_refinement_trace_sha256,
        empirical_remainder_claim_ids=tuple(
            str(item) for item in empirical_remainder_claim_ids
        ),
        notes=notes,
    )
    return build_formal_empirical_split(
        formal_status=formal_status,
        empirical_status=empirical_status,
        refs=refs,
        empirical_remainder_claim_ids=empirical_remainder_claim_ids,
        decision_support=support,
    )


def validate_formal_empirical_separation(
    *,
    formal_status: FormalEvidenceStatus | str,
    empirical_status: EmpiricalEvidenceStatus | str,
    ship_gates_passed: bool,
    claim_class: str,
    exploratory: bool = False,
) -> tuple[str, ...]:
    """Fail closed when formal evidence is treated as empirical promotion authority."""

    failures: list[str] = []
    # Proof / strong formal cannot alone authorize ship or promotion.
    if formal_status in _STRONG_FORMAL and empirical_status != "success":
        if ship_gates_passed:
            failures.append("formal_cannot_authorize_ship_gates")
        if claim_class in {"promotion_candidate", "ship_gate"} and not exploratory:
            # Not an automatic failure of the *result payload* — promotion
            # still goes through validate_result_claim empirical path — but
            # record the separation when someone stamps ship_gates_passed.
            pass
    if empirical_status != "success" and ship_gates_passed:
        # Empirical failure/inconclusive with ship stamp is always wrong,
        # regardless of formal status (proved preflight + failed empirical).
        failures.append("empirical_status_blocks_ship_gates")
    if (
        formal_status in _WEAK_FORMAL
        and empirical_status == "success"
        and claim_class in {"promotion_candidate", "ship_gate"}
    ):
        # Allowed: weak/unknown formal + successful empirical. No failure.
        pass
    return tuple(failures)


def decision_support_report(
    split: CampaignFormalEmpiricalSplitV1 | Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Query surface: assumptions / bounds / refinement backing a decision."""

    if split is None:
        return {
            "schema": "campaign_decision_support_report/v1",
            "present": False,
            "formal_status": "absent",
            "empirical_status": "not_run",
            "refs": [],
            "axes": {},
            "empirical_remainder_claim_ids": [],
            "notes": "no formal_empirical split on campaign result (historical or unset)",
        }
    if not isinstance(split, CampaignFormalEmpiricalSplitV1):
        split = CampaignFormalEmpiricalSplitV1.model_validate(split)
    ds = split.decision_support
    axes = {
        "assumption_strength": None if ds is None else ds.assumption_axis_status,
        "computability": None if ds is None else ds.computability_axis_status,
        "resource_bounds": None if ds is None else ds.resource_bounds_axis_status,
        "implementation_refinement": None if ds is None else ds.refinement_axis_status,
    }
    return {
        "schema": "campaign_decision_support_report/v1",
        "present": True,
        "formal_status": split.formal_status,
        "empirical_status": split.empirical_status,
        "content_digest": split.content_digest,
        "refs": [ref.model_dump(mode="json") for ref in split.refs],
        "axes": axes,
        "bound_ast_ids": [] if ds is None else list(ds.bound_ast_ids),
        "formal_authority_digests": (
            [] if ds is None else list(ds.formal_authority_digests)
        ),
        "four_axis_ledger_sha256": None if ds is None else ds.four_axis_ledger_sha256,
        "revmath_report_sha256": None if ds is None else ds.revmath_report_sha256,
        "runtime_refinement_trace_sha256": (
            None if ds is None else ds.runtime_refinement_trace_sha256
        ),
        "empirical_remainder_claim_ids": list(split.empirical_remainder_claim_ids),
        "notes": "" if ds is None else ds.notes,
        "separation": {
            "formal_authorizes_empirical": False,
            "strong_formal": split.formal_status in _STRONG_FORMAL,
            "weak_or_unknown_formal": split.formal_status in _WEAK_FORMAL,
            "empirical_success": split.empirical_status == "success",
        },
    }


def migrate_campaign_result_payload(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Mixed-version reader: historical payloads keep ``formal_empirical`` unset.

    Never invents digests, axis statuses, or empirical remainders.
    """

    payload = dict(raw)
    if "formal_empirical" not in payload:
        # Leave absent so model defaults to None (exact historical shape on dump
        # when using exclude_none; validation accepts missing field).
        return payload
    value = payload.get("formal_empirical")
    if value is None:
        return payload
    if not isinstance(value, Mapping):
        raise ValueError("formal_empirical must be an object or null")
    # Re-seal digest if older writers omitted it but body is otherwise complete.
    if "content_digest" not in value:
        body = dict(value)
        payload["formal_empirical"] = {
            **body,
            "content_digest": _split_content_digest(body),
        }
    else:
        # Round-trip validate; raises on tamper.
        CampaignFormalEmpiricalSplitV1.model_validate(value)
    return payload


def attach_formal_empirical_to_result_payload(
    result_payload: Mapping[str, Any],
    split: CampaignFormalEmpiricalSplitV1 | Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return a result dict with optional split attached (None clears)."""

    out = dict(result_payload)
    if split is None:
        out.pop("formal_empirical", None)
        return out
    if isinstance(split, CampaignFormalEmpiricalSplitV1):
        out["formal_empirical"] = split.model_dump(mode="json")
    else:
        out["formal_empirical"] = dict(split)
    return migrate_campaign_result_payload(out)


def ledger_axis_statuses(
    ledger: Mapping[str, Any] | Any | None,
) -> dict[str, str | None]:
    """Pull four-axis statuses from a FormalPreflightFourAxisLedgerV1-shaped object."""

    if ledger is None:
        return {
            "assumption_strength": None,
            "computability": None,
            "resource_bounds": None,
            "implementation_refinement": None,
        }
    if hasattr(ledger, "model_dump"):
        data = ledger.model_dump(mode="json")
    else:
        data = dict(ledger)
    analysis = data.get("analysis") or {}
    if hasattr(analysis, "model_dump"):
        analysis = analysis.model_dump(mode="json")

    def _status(axis_key: str) -> str | None:
        axis = analysis.get(axis_key) or {}
        if hasattr(axis, "model_dump"):
            axis = axis.model_dump(mode="json")
        status = axis.get("status")
        return None if status is None else str(status)

    return {
        "assumption_strength": _status("assumption_strength"),
        "computability": _status("computability"),
        "resource_bounds": _status("resource_bounds"),
        "implementation_refinement": _status("implementation_refinement"),
    }


def four_axis_ledger_digest(ledger: Mapping[str, Any] | Any) -> str:
    """Content digest for a four-axis ledger payload (EVID-03)."""

    if hasattr(ledger, "model_dump"):
        payload = ledger.model_dump(mode="json")
    else:
        payload = dict(ledger)
    return content_sha(payload)


def revmath_report_digest(report: Mapping[str, Any] | Any) -> str:
    if hasattr(report, "model_dump"):
        payload = report.model_dump(mode="json")
    else:
        payload = dict(report)
    return content_sha(payload)


__all__ = [
    "CONTENT_BOUND_REF_SCHEMA",
    "CampaignDecisionSupportV1",
    "CampaignFormalEmpiricalSplitV1",
    "ContentBoundEvidenceRefV1",
    "DECISION_SUPPORT_SCHEMA",
    "EmpiricalEvidenceStatus",
    "EvidenceRefKind",
    "FORMAL_EMPIRICAL_SPLIT_SCHEMA",
    "FormalEvidenceStatus",
    "attach_formal_empirical_to_result_payload",
    "build_formal_empirical_split",
    "decision_support_report",
    "empirical_status_from_result",
    "formal_status_from_authority",
    "four_axis_ledger_digest",
    "ledger_axis_statuses",
    "link_four_axis_and_revmath",
    "migrate_campaign_result_payload",
    "revmath_report_digest",
    "validate_formal_empirical_separation",
]
