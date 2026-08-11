"""Formal claim/evidence authority envelope (EVID-06 / SLM-526).

Converges portable ``FormalObjectV1`` and campaign ``FormalPreflightV1`` into
one ``formal_authority/v2`` contract via adapters — not a third evidence stack,
proof runner, or campaign store. Historical v1 artifacts stay readable;
preferred writes emit v2.

Incomplete / skipped / unknown evidence is structurally incapable of being
represented as checked semantic refutation (reuses KERN-01 ``SolverJudgmentV1``).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from slm_training.formal.judgment import (
    JudgmentOutcome,
    SolverJudgmentV1,
    UnknownReason,
    from_formal_preflight_status,
)
from slm_training.formal.objects import (
    FORMAL_OBJECT_SCHEMA,
    FormalObjectV1,
)

FORMAL_AUTHORITY_SCHEMA = "formal_authority/v2"
FORMAL_PREFLIGHT_SCHEMA = "FormalPreflightV1"

AuthoritySurface = Literal["formal_object", "formal_preflight"]
AuthorityClassName = Literal[
    "checked_semantic",
    "unchecked",
    "invalid",
    "empirical_remainder",
]
CheckerRunStatus = Literal["pass", "fail", "skipped", "unknown"]

SUPPORTED_READ_SCHEMAS: frozenset[str] = frozenset(
    {FORMAL_AUTHORITY_SCHEMA, FORMAL_OBJECT_SCHEMA, FORMAL_PREFLIGHT_SCHEMA}
)
PREFERRED_WRITE_SCHEMA = FORMAL_AUTHORITY_SCHEMA

_INCOMPLETE_CHECKER_STATUSES: frozenset[str] = frozenset({"skipped", "unknown"})
_UNCHECKED_PREFLIGHT_STATUSES: frozenset[str] = frozenset(
    {"unknown", "timed_out", "conditional"}
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CheckerCapabilityRefV1:
    backend: str
    available: bool
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "available": self.available,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CheckerCapabilityRefV1:
        return cls(
            backend=str(data["backend"]),
            available=bool(data["available"]),
            notes=str(data.get("notes", "")),
        )


@dataclass(frozen=True)
class CheckerResultRefV1:
    backend: str
    status: CheckerRunStatus
    detail: str = ""
    result_digest: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "backend": self.backend,
            "status": self.status,
            "detail": self.detail,
        }
        if self.result_digest is not None:
            out["result_digest"] = self.result_digest
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CheckerResultRefV1:
        status = str(data["status"])
        if status not in {"pass", "fail", "skipped", "unknown"}:
            raise ValueError(f"invalid checker status: {status!r}")
        digest = data.get("result_digest")
        return cls(
            backend=str(data["backend"]),
            status=status,  # type: ignore[arg-type]
            detail=str(data.get("detail", "")),
            result_digest=None if digest is None else str(digest),
        )


@dataclass(frozen=True)
class ToolchainIdentityV1:
    lean_version: str | None = None
    mathlib_version: str | None = None
    build_output_sha256: str | None = None
    checker_contract: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "lean_version": self.lean_version,
            "mathlib_version": self.mathlib_version,
            "build_output_sha256": self.build_output_sha256,
            "checker_contract": self.checker_contract,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> ToolchainIdentityV1:
        if not data:
            return cls()
        build = data.get("build_output_sha256")
        return cls(
            lean_version=None if data.get("lean_version") is None else str(data["lean_version"]),
            mathlib_version=(
                None
                if data.get("mathlib_version") is None
                else str(data["mathlib_version"])
            ),
            build_output_sha256=None if build is None else str(build),
            checker_contract=(
                None
                if data.get("checker_contract") is None
                else str(data["checker_contract"])
            ),
        )


def derive_authority_class(
    judgment: SolverJudgmentV1,
    *,
    empirical_remainder: bool = False,
) -> AuthorityClassName:
    """Derive authority class from KERN-01 judgment (never from vibes)."""

    if judgment.outcome is JudgmentOutcome.INVALID:
        return "invalid"
    if judgment.authorizes_semantic_conclusion():
        return "checked_semantic"
    if empirical_remainder:
        return "empirical_remainder"
    return "unchecked"


def _has_incomplete_checker_evidence(
    checker_results: Sequence[CheckerResultRefV1],
) -> bool:
    return any(item.status in _INCOMPLETE_CHECKER_STATUSES for item in checker_results)


def _validate_authority_invariants(
    *,
    judgment: SolverJudgmentV1,
    authority_class: AuthorityClassName,
    checker_results: Sequence[CheckerResultRefV1],
    empirical_remainder_claim_ids: Sequence[str],
) -> None:
    if not judgment.payload_matches_outcome():
        raise ValueError("judgment status tag / payload mismatch")

    derived = derive_authority_class(
        judgment,
        empirical_remainder=bool(empirical_remainder_claim_ids)
        and not judgment.authorizes_semantic_conclusion(),
    )
    # Explicit empirical_remainder is allowed only when semantic authority is absent.
    if authority_class == "empirical_remainder":
        if judgment.authorizes_semantic_conclusion():
            raise ValueError(
                "empirical_remainder cannot carry checked semantic authority"
            )
        if judgment.outcome is JudgmentOutcome.INVALID:
            raise ValueError("invalid judgment cannot be empirical_remainder")
    elif authority_class != derived:
        raise ValueError(
            f"authority_class {authority_class!r} contradicts judgment "
            f"(derived={derived!r})"
        )

    if authority_class == "checked_semantic":
        if not judgment.authorizes_semantic_conclusion():
            raise ValueError(
                "checked_semantic requires checked witnessed/refuted judgment"
            )
        if _has_incomplete_checker_evidence(checker_results):
            raise ValueError(
                "incomplete/skipped/unknown checker evidence cannot be "
                "checked semantic authority"
            )
        if judgment.outcome is JudgmentOutcome.REFUTED and not judgment.checked:
            raise ValueError(
                "unchecked refutation cannot be checked semantic refutation"
            )

    if judgment.outcome is JudgmentOutcome.REFUTED and judgment.checked:
        if _has_incomplete_checker_evidence(checker_results):
            raise ValueError(
                "incomplete/skipped/unknown evidence cannot be represented as "
                "checked semantic refutation"
            )
        if isinstance(judgment.payload, UnknownReason):
            raise ValueError("unknown payload cannot authorize semantic refutation")


def _authority_digest_body(data: Mapping[str, Any]) -> dict[str, Any]:
    body = {key: value for key, value in data.items() if key != "content_digest"}
    return body


@dataclass(frozen=True)
class FormalAuthorityV2:
    """Single v2 claim/evidence authority envelope (EVID-06)."""

    claim_id: str
    surface: AuthoritySurface
    judgment: SolverJudgmentV1
    authority_class: AuthorityClassName
    v1_artifact: dict[str, Any]
    v1_content_digest: str
    four_axis_ledger_sha256: str | None
    bound_ast_ids: tuple[str, ...]
    checker_results: tuple[CheckerResultRefV1, ...]
    checker_capabilities: tuple[CheckerCapabilityRefV1, ...]
    source_digests: dict[str, str]
    toolchain: ToolchainIdentityV1
    certificate_digest: str | None
    replay_digest: str | None
    empirical_remainder_claim_ids: tuple[str, ...]
    empirical_remainder_notes: str
    content_digest: str
    schema: str = FORMAL_AUTHORITY_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "claim_id": self.claim_id,
            "surface": self.surface,
            "judgment": self.judgment.to_dict(),
            "authority_class": self.authority_class,
            "v1_artifact": self.v1_artifact,
            "v1_content_digest": self.v1_content_digest,
            "four_axis_ledger_sha256": self.four_axis_ledger_sha256,
            "bound_ast_ids": list(self.bound_ast_ids),
            "checker_results": [item.to_dict() for item in self.checker_results],
            "checker_capabilities": [
                item.to_dict() for item in self.checker_capabilities
            ],
            "source_digests": dict(self.source_digests),
            "toolchain": self.toolchain.to_dict(),
            "certificate_digest": self.certificate_digest,
            "replay_digest": self.replay_digest,
            "empirical_remainder_claim_ids": list(self.empirical_remainder_claim_ids),
            "empirical_remainder_notes": self.empirical_remainder_notes,
            "content_digest": self.content_digest,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FormalAuthorityV2:
        if data.get("schema") != FORMAL_AUTHORITY_SCHEMA:
            raise ValueError(
                f"expected schema {FORMAL_AUTHORITY_SCHEMA!r}, got {data.get('schema')!r}"
            )
        surface = str(data["surface"])
        if surface not in {"formal_object", "formal_preflight"}:
            raise ValueError(f"unknown authority surface: {surface!r}")
        authority_class = str(data["authority_class"])
        if authority_class not in {
            "checked_semantic",
            "unchecked",
            "invalid",
            "empirical_remainder",
        }:
            raise ValueError(f"unknown authority_class: {authority_class!r}")
        judgment = SolverJudgmentV1.from_dict(dict(data["judgment"]))
        checker_results = tuple(
            CheckerResultRefV1.from_dict(item)
            for item in data.get("checker_results", ())
        )
        checker_capabilities = tuple(
            CheckerCapabilityRefV1.from_dict(item)
            for item in data.get("checker_capabilities", ())
        )
        empirical_ids = tuple(
            str(item) for item in data.get("empirical_remainder_claim_ids", ())
        )
        _validate_authority_invariants(
            judgment=judgment,
            authority_class=authority_class,  # type: ignore[arg-type]
            checker_results=checker_results,
            empirical_remainder_claim_ids=empirical_ids,
        )
        obj = cls(
            claim_id=str(data["claim_id"]),
            surface=surface,  # type: ignore[arg-type]
            judgment=judgment,
            authority_class=authority_class,  # type: ignore[arg-type]
            v1_artifact=dict(data["v1_artifact"]),
            v1_content_digest=str(data["v1_content_digest"]),
            four_axis_ledger_sha256=(
                None
                if data.get("four_axis_ledger_sha256") is None
                else str(data["four_axis_ledger_sha256"])
            ),
            bound_ast_ids=tuple(str(item) for item in data.get("bound_ast_ids", ())),
            checker_results=checker_results,
            checker_capabilities=checker_capabilities,
            source_digests={
                str(key): str(value)
                for key, value in dict(data.get("source_digests", {})).items()
            },
            toolchain=ToolchainIdentityV1.from_dict(data.get("toolchain")),
            certificate_digest=(
                None
                if data.get("certificate_digest") is None
                else str(data["certificate_digest"])
            ),
            replay_digest=(
                None
                if data.get("replay_digest") is None
                else str(data["replay_digest"])
            ),
            empirical_remainder_claim_ids=empirical_ids,
            empirical_remainder_notes=str(data.get("empirical_remainder_notes", "")),
            content_digest=str(data["content_digest"]),
            schema=str(data["schema"]),
        )
        recomputed = _sha256(_canonical_json(_authority_digest_body(obj.to_dict())))
        if recomputed != obj.content_digest:
            raise ValueError(
                "formal authority content_digest mismatch "
                f"(declared={obj.content_digest}, recomputed={recomputed})"
            )
        return obj


def build_formal_authority_v2(
    *,
    claim_id: str,
    surface: AuthoritySurface,
    judgment: SolverJudgmentV1,
    v1_artifact: Mapping[str, Any],
    v1_content_digest: str,
    four_axis_ledger_sha256: str | None = None,
    bound_ast_ids: Sequence[str] = (),
    checker_results: Sequence[CheckerResultRefV1] = (),
    checker_capabilities: Sequence[CheckerCapabilityRefV1] = (),
    source_digests: Mapping[str, str] | None = None,
    toolchain: ToolchainIdentityV1 | None = None,
    certificate_digest: str | None = None,
    replay_digest: str | None = None,
    empirical_remainder_claim_ids: Sequence[str] = (),
    empirical_remainder_notes: str = "",
    authority_class: AuthorityClassName | None = None,
) -> FormalAuthorityV2:
    """Construct a validated v2 envelope (preferred write path)."""

    results = tuple(checker_results)
    empirical_ids = tuple(str(item) for item in empirical_remainder_claim_ids)
    derived = derive_authority_class(
        judgment,
        empirical_remainder=bool(empirical_ids)
        or bool(empirical_remainder_notes),
    )
    klass: AuthorityClassName = authority_class or derived
    _validate_authority_invariants(
        judgment=judgment,
        authority_class=klass,
        checker_results=results,
        empirical_remainder_claim_ids=empirical_ids,
    )
    provisional = {
        "schema": FORMAL_AUTHORITY_SCHEMA,
        "claim_id": claim_id,
        "surface": surface,
        "judgment": judgment.to_dict(),
        "authority_class": klass,
        "v1_artifact": dict(v1_artifact),
        "v1_content_digest": v1_content_digest,
        "four_axis_ledger_sha256": four_axis_ledger_sha256,
        "bound_ast_ids": list(bound_ast_ids),
        "checker_results": [item.to_dict() for item in results],
        "checker_capabilities": [
            item.to_dict() for item in checker_capabilities
        ],
        "source_digests": dict(source_digests or {}),
        "toolchain": (toolchain or ToolchainIdentityV1()).to_dict(),
        "certificate_digest": certificate_digest,
        "replay_digest": replay_digest,
        "empirical_remainder_claim_ids": list(empirical_ids),
        "empirical_remainder_notes": empirical_remainder_notes,
    }
    digest = _sha256(_canonical_json(provisional))
    return FormalAuthorityV2.from_dict({**provisional, "content_digest": digest})


def checker_result_from_backend(
    *,
    backend: str,
    ok: bool,
    detail: str = "",
    skipped: bool = False,
    unknown: bool = False,
    result_digest: str | None = None,
) -> CheckerResultRefV1:
    if unknown:
        status: CheckerRunStatus = "unknown"
    elif skipped:
        status = "skipped"
    elif ok:
        status = "pass"
    else:
        status = "fail"
    return CheckerResultRefV1(
        backend=backend,
        status=status,
        detail=detail,
        result_digest=result_digest,
    )


def from_formal_object_v1(
    obj: FormalObjectV1,
    *,
    judgment: SolverJudgmentV1,
    checker_results: Sequence[CheckerResultRefV1] = (),
    checker_capabilities: Sequence[CheckerCapabilityRefV1] = (),
    certificate_digest: str | None = None,
    replay_digest: str | None = None,
    empirical_remainder_claim_ids: Sequence[str] = (),
    empirical_remainder_notes: str = "",
) -> FormalAuthorityV2:
    """Narrow adapter: portable FormalObjectV1 → authority v2."""

    artifact = obj.to_dict()
    cert = certificate_digest
    if cert is None and isinstance(obj.statement, dict):
        maybe = obj.statement.get("certificate_digest")
        if maybe is not None:
            cert = str(maybe)
    capabilities = tuple(checker_capabilities)
    if not capabilities:
        capabilities = tuple(
            CheckerCapabilityRefV1(backend=name, available=True)
            for name in obj.required_checkers
        )
    return build_formal_authority_v2(
        claim_id=obj.object_id,
        surface="formal_object",
        judgment=judgment,
        v1_artifact=artifact,
        v1_content_digest=obj.content_digest,
        checker_results=checker_results,
        checker_capabilities=capabilities,
        source_digests={"formal_object.content_digest": obj.content_digest},
        certificate_digest=cert,
        replay_digest=replay_digest,
        empirical_remainder_claim_ids=empirical_remainder_claim_ids,
        empirical_remainder_notes=empirical_remainder_notes,
    )


def to_formal_object_v1(authority: FormalAuthorityV2) -> FormalObjectV1:
    """Downgrade adapter: authority v2 → FormalObjectV1 (lossless via projection)."""

    if authority.surface != "formal_object":
        raise ValueError(
            f"cannot downgrade surface {authority.surface!r} to FormalObjectV1"
        )
    return FormalObjectV1.from_dict(authority.v1_artifact)


def from_formal_preflight_v1(
    preflight: Any,
    *,
    judgment: SolverJudgmentV1 | None = None,
    checker_results: Sequence[CheckerResultRefV1] = (),
    checker_capabilities: Sequence[CheckerCapabilityRefV1] = (),
    four_axis_ledger_sha256: str | None = None,
    bound_ast_ids: Sequence[str] = (),
    v1_content_digest: str | None = None,
) -> FormalAuthorityV2:
    """Narrow adapter: FormalPreflightV1 → authority v2.

    Incomplete preflight statuses (unknown/timed_out/conditional) always map to
    an unchecked UNKNOWN judgment — never checked semantic refutation.
    """

    from slm_training.autoresearch.formal import (
        formal_preflight_payload,
        formal_preflight_sha256,
    )
    from slm_training.autoresearch.schemas import FormalPreflightV1

    if not isinstance(preflight, FormalPreflightV1):
        raise TypeError("from_formal_preflight_v1 requires FormalPreflightV1")

    status = str(preflight.status)
    checked = status == "proved" or (
        status == "refuted" and preflight.counterexample is not None
    )
    if status in _UNCHECKED_PREFLIGHT_STATUSES:
        # Structural: incomplete lifecycle never becomes checked refutation.
        # Caller-supplied judgments are ignored for these statuses.
        mapped = from_formal_preflight_status(
            status,
            counterexample=preflight.counterexample,
            checked=False,
        )
    else:
        mapped = judgment or from_formal_preflight_status(
            status,
            counterexample=preflight.counterexample,
            checked=checked,
        )
    if (
        mapped.outcome is JudgmentOutcome.REFUTED
        and mapped.checked
        and (
            status in _UNCHECKED_PREFLIGHT_STATUSES
            or _has_incomplete_checker_evidence(tuple(checker_results))
        )
    ):
        raise ValueError(
            "incomplete/skipped/unknown evidence cannot be represented as "
            "checked semantic refutation"
        )

    ledger = preflight.four_axis_ledger
    ledger_sha = four_axis_ledger_sha256
    empirical_ids: tuple[str, ...] = ()
    empirical_notes = ""
    bound_ids = tuple(bound_ast_ids)
    if ledger is not None:
        empirical_ids = tuple(ledger.empirical_remainder_claim_ids)
        if ledger.analysis.implementation_refinement.notes:
            empirical_notes = str(ledger.analysis.implementation_refinement.notes)
        rb = ledger.analysis.resource_bounds
        if rb.bound_ast_id is not None and rb.bound_ast_id not in bound_ids:
            bound_ids = (*bound_ids, str(rb.bound_ast_id))
        if ledger_sha is None:
            ledger_sha = _sha256(
                _canonical_json(ledger.model_dump(mode="json"))
            )

    artifact = formal_preflight_payload(preflight)
    digest = v1_content_digest or formal_preflight_sha256(preflight)
    results = tuple(checker_results)
    if not results and preflight.checker_contract:
        # Lifecycle-only: no checker run recorded → unknown, not a pass.
        results = (
            CheckerResultRefV1(
                backend="lean_kernel",
                status="unknown" if status in _UNCHECKED_PREFLIGHT_STATUSES else "pass",
                detail=str(preflight.checker_contract),
            ),
        )

    capabilities = tuple(checker_capabilities)
    if not capabilities:
        capabilities = (
            CheckerCapabilityRefV1(
                backend="lean_kernel",
                available=status not in {"timed_out", "unknown"},
                notes=str(preflight.checker_contract or ""),
            ),
        )

    return build_formal_authority_v2(
        claim_id=preflight.obligation_id,
        surface="formal_preflight",
        judgment=mapped,
        v1_artifact=artifact,
        v1_content_digest=digest,
        four_axis_ledger_sha256=ledger_sha,
        bound_ast_ids=bound_ids,
        checker_results=results,
        checker_capabilities=capabilities,
        source_digests=dict(preflight.source_digests),
        toolchain=ToolchainIdentityV1(
            lean_version=preflight.lean_version,
            mathlib_version=preflight.mathlib_version,
            build_output_sha256=preflight.build_output_sha256,
            checker_contract=preflight.checker_contract,
        ),
        certificate_digest=preflight.proof_sha256,
        empirical_remainder_claim_ids=empirical_ids,
        empirical_remainder_notes=empirical_notes,
    )


def to_formal_preflight_v1(authority: FormalAuthorityV2) -> Any:
    """Downgrade adapter: authority v2 → FormalPreflightV1."""

    if authority.surface != "formal_preflight":
        raise ValueError(
            f"cannot downgrade surface {authority.surface!r} to FormalPreflightV1"
        )
    from slm_training.autoresearch.formal import migrate_formal_preflight_v1

    return migrate_formal_preflight_v1(authority.v1_artifact)


def negotiate_schema(requested: str | None) -> str:
    """Version negotiation: unknown schemas fail closed; default write is v2."""

    if requested is None or requested == "":
        return PREFERRED_WRITE_SCHEMA
    if requested not in SUPPORTED_READ_SCHEMAS:
        raise ValueError(
            f"unsupported formal authority schema {requested!r}; "
            f"supported={sorted(SUPPORTED_READ_SCHEMAS)}"
        )
    return requested


def detect_payload_schema(payload: Mapping[str, Any]) -> str:
    if "schema" in payload and payload["schema"] in SUPPORTED_READ_SCHEMAS:
        return str(payload["schema"])
    schema_version = payload.get("schema_version")
    if schema_version in SUPPORTED_READ_SCHEMAS:
        return str(schema_version)
    raise ValueError(
        "unable to detect formal authority schema; expected one of "
        f"{sorted(SUPPORTED_READ_SCHEMAS)}"
    )


def load_formal_authority(
    payload: Mapping[str, Any],
    *,
    judgment: SolverJudgmentV1 | None = None,
    checker_results: Sequence[CheckerResultRefV1] = (),
) -> FormalAuthorityV2:
    """Read v1 or v2 payloads into the v2 authority envelope (fail closed)."""

    schema = detect_payload_schema(payload)
    negotiate_schema(schema)
    if schema == FORMAL_AUTHORITY_SCHEMA:
        return FormalAuthorityV2.from_dict(payload)
    if schema == FORMAL_OBJECT_SCHEMA:
        obj = FormalObjectV1.from_dict(payload)
        if judgment is None:
            raise ValueError(
                "loading formal_object/v1 into authority v2 requires an explicit "
                "SolverJudgmentV1 (adapters never invent semantic judgment)"
            )
        return from_formal_object_v1(
            obj,
            judgment=judgment,
            checker_results=checker_results,
        )
    from slm_training.autoresearch.formal import migrate_formal_preflight_v1

    preflight = migrate_formal_preflight_v1(payload)
    return from_formal_preflight_v1(
        preflight,
        judgment=judgment,
        checker_results=checker_results,
    )


def prefer_write_payload(authority: FormalAuthorityV2) -> dict[str, Any]:
    """Preferred write path: always emit formal_authority/v2."""

    if authority.schema != FORMAL_AUTHORITY_SCHEMA:
        raise ValueError("prefer_write_payload requires formal_authority/v2")
    return authority.to_dict()


__all__ = [
    "FORMAL_AUTHORITY_SCHEMA",
    "FORMAL_PREFLIGHT_SCHEMA",
    "PREFERRED_WRITE_SCHEMA",
    "SUPPORTED_READ_SCHEMAS",
    "AuthorityClassName",
    "AuthoritySurface",
    "CheckerCapabilityRefV1",
    "CheckerResultRefV1",
    "FormalAuthorityV2",
    "ToolchainIdentityV1",
    "build_formal_authority_v2",
    "checker_result_from_backend",
    "derive_authority_class",
    "detect_payload_schema",
    "from_formal_object_v1",
    "from_formal_preflight_v1",
    "load_formal_authority",
    "negotiate_schema",
    "prefer_write_payload",
    "to_formal_object_v1",
    "to_formal_preflight_v1",
]
