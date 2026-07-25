"""Marker-table materialization with alpha-renaming and split-level permutation.

SLM-356 (DSH1-04). A marker table projects the request's runtime symbols onto
opaque, request-local marker surfaces plus *declared* typed authority
(type/role/scope/signature). The logical-to-surface mapping lives only in a
sidecar provenance record, never in model-facing rows. Ordinal assignment is
randomized per (root_family, split) with independent seeds so a marker's role
is not recoverable from its ordinal and held-out surfaces are disjoint from
train surfaces.

Fail-closed audits cover exact-copy leakage, alpha-renaming identity,
unknown / duplicate / out-of-range marker use, and semantic surface
dependence (the corpus-publication stop rule).
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Literal

from slm_training.data.contract import GenerationRequest, RuntimeSymbol
from slm_training.harness_core.lineage.records import content_sha
from slm_training.harnesses.train_data.split_policy import RootFamilySplitPolicyV1

SCHEMA_VERSION = "marker_table/v1"
PROVENANCE_SCHEMA_VERSION = "marker_table_provenance/v1"
POLICY_VERSION = "marker_table_policy/v1"

MarkerRole = Literal["binder", "external_entity", "state_ref", "fresh_binder"]
MARKER_ROLES: tuple[MarkerRole, ...] = (
    "binder",
    "external_entity",
    "state_ref",
    "fresh_binder",
)

# RuntimeSymbol role <-> marker-table role. RuntimeSymbol surface codecs are
# preserved (external ':', state '$') so tables stay compatible with the V8
# dynamic-vocabulary path; the codec prefix is declared authority, while the
# ordinal carries no role information.
_RUNTIME_TO_MARKER_ROLE = {
    "alpha_binder": "binder",
    "external_entity": "external_entity",
    "state": "state_ref",
    "fresh_binder": "fresh_binder",
}
_MARKER_TO_RUNTIME_ROLE = {value: key for key, value in _RUNTIME_TO_MARKER_ROLE.items()}

# Typed facts that are inference-visible on a model-facing row. ``description``
# is provenance-only (free text may embed the logical name).
_INFERENCE_VISIBLE_FIELDS = ("semantic_type", "semantic_role", "scope", "signature")


class MarkerAuditCode(str, Enum):
    EXACT_COPY = "marker_exact_copy"
    SEMANTIC_SURFACE = "marker_semantic_surface_dependence"
    ALPHA_MISMATCH = "marker_alpha_mismatch"
    UNKNOWN = "marker_unknown"
    DUPLICATE = "marker_duplicate"
    OUT_OF_RANGE = "marker_out_of_range"
    ORDINAL_ROLE_LEAK = "marker_ordinal_role_leak"


@dataclass(frozen=True)
class MarkerRowV1:
    """One model-facing marker row: opaque surface + declared typed authority."""

    surface: str
    role: MarkerRole
    ordinal: int
    semantic_type: str | None = None
    semantic_role: str | None = None
    scope: str | None = None
    signature: str | None = None

    def __post_init__(self) -> None:
        if self.role not in MARKER_ROLES:
            raise ValueError(f"unknown marker role {self.role!r}")
        if not self.surface:
            raise ValueError("marker surface must be non-empty")
        _require_surface_codec(self.surface, self.role)
        if self.ordinal < 0:
            raise ValueError("marker ordinal must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "surface": self.surface,
            "role": self.role,
            "ordinal": self.ordinal,
        }
        for key in _INFERENCE_VISIBLE_FIELDS:
            value = getattr(self, key)
            if value is not None:
                data[key] = value
        return data

    def to_runtime_symbol(self) -> RuntimeSymbol:
        """Project onto the V8 ``RuntimeSymbol`` contract (fail closed)."""
        return RuntimeSymbol(
            surface=self.surface,
            role=_MARKER_TO_RUNTIME_ROLE[self.role],  # type: ignore[arg-type]
            semantic_type=self.semantic_type,
            semantic_role=self.semantic_role,
            scope=self.scope,
            signature=self.signature,
        )


@dataclass(frozen=True)
class MarkerTableV1:
    """Model-facing marker table. Contains no logical names."""

    table_id: str
    root_family_id: str
    split: str
    policy_version: str
    permutation_digest: str
    rows: tuple[MarkerRowV1, ...]
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported marker table schema {self.schema_version!r}")
        if self.policy_version != POLICY_VERSION:
            raise ValueError(f"unsupported marker policy {self.policy_version!r}")
        _require_digest(self.table_id, "table_id")
        _require_digest(self.permutation_digest, "permutation_digest")
        RootFamilySplitPolicyV1().require_inherited(
            root_family_id=self.root_family_id,
            split_group_id=self.root_family_id,
            split=self.split,
        )
        surfaces = [row.surface for row in self.rows]
        if len(surfaces) != len(set(surfaces)):
            raise ValueError("duplicate marker surface in table")
        ordinals = [row.ordinal for row in self.rows]
        if len(ordinals) != len(set(ordinals)):
            raise ValueError("duplicate marker ordinal in table")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "table_id": self.table_id,
            "root_family_id": self.root_family_id,
            "split": self.split,
            "policy_version": self.policy_version,
            "permutation_digest": self.permutation_digest,
            "rows": [row.to_dict() for row in self.rows],
        }

    def surfaces(self) -> tuple[str, ...]:
        return tuple(row.surface for row in self.rows)

    def runtime_symbols(self) -> tuple[RuntimeSymbol, ...]:
        return tuple(row.to_runtime_symbol() for row in self.rows)


@dataclass(frozen=True)
class MarkerProvenanceEntryV1:
    """Sidecar-only logical-to-surface mapping. Never model-facing."""

    logical_surface: str
    marker_surface: str

    def to_dict(self) -> dict[str, str]:
        return {
            "logical_surface": self.logical_surface,
            "marker_surface": self.marker_surface,
        }


@dataclass(frozen=True)
class MarkerTableProvenanceV1:
    table_id: str
    entries: tuple[MarkerProvenanceEntryV1, ...]
    master_seed: int
    schema_version: str = PROVENANCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PROVENANCE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported provenance schema {self.schema_version!r}"
            )
        _require_digest(self.table_id, "table_id")
        logicals = [entry.logical_surface for entry in self.entries]
        markers = [entry.marker_surface for entry in self.entries]
        if len(logicals) != len(set(logicals)):
            raise ValueError("duplicate logical surface in provenance")
        if len(markers) != len(set(markers)):
            raise ValueError("duplicate marker surface in provenance")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "table_id": self.table_id,
            "master_seed": self.master_seed,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    def surface_for(self, logical_surface: str) -> str:
        for entry in self.entries:
            if entry.logical_surface == logical_surface:
                return entry.marker_surface
        raise KeyError(f"unknown logical surface {logical_surface!r}")


@dataclass(frozen=True)
class MarkerTableBundleV1:
    table: MarkerTableV1
    provenance: MarkerTableProvenanceV1


@dataclass(frozen=True)
class MarkerAuditFindingV1:
    code: MarkerAuditCode
    surface: str
    evidence: str

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code.value,
            "surface": self.surface,
            "evidence": self.evidence,
        }


def split_permutation_seed(
    master_seed: int,
    root_family_id: str,
    split: str,
    *,
    policy_version: str = POLICY_VERSION,
) -> int:
    """Independent deterministic seed per (root_family, split)."""
    digest = hashlib.sha256(
        f"{policy_version}|{master_seed}|{root_family_id}|{split}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big")


def materialize_marker_table(
    request: GenerationRequest,
    *,
    root_family_id: str,
    split: str,
    master_seed: int,
    policy_version: str = POLICY_VERSION,
) -> MarkerTableBundleV1:
    """Materialize opaque marker surfaces for a request's runtime symbols.

    Only inference-visible typed facts (type/role/scope/signature) are copied
    onto model-facing rows; ``description`` and the logical surface stay in
    provenance. Ordinals are assigned by a deranged permutation seeded
    independently per (root_family, split).
    """
    symbols = request.effective_runtime_symbols()
    seed = split_permutation_seed(
        master_seed, root_family_id, split, policy_version=policy_version
    )
    return _materialize(
        symbols,
        root_family_id=root_family_id,
        split=split,
        seed=seed,
        master_seed=master_seed,
        policy_version=policy_version,
    )


def permute_marker_surfaces(
    bundle: MarkerTableBundleV1,
    *,
    seed: int,
) -> MarkerTableBundleV1:
    """Alpha-rename a table: fresh opaque surfaces, identical logical content.

    The canonical (surface-free) payload is unchanged, so canonical AST
    identity is invariant under this permutation.
    """
    table = bundle.table
    logical_by_surface = {
        entry.marker_surface: entry.logical_surface
        for entry in bundle.provenance.entries
    }
    symbols = tuple(
        _symbol_for_row(row, logical_by_surface[row.surface]) for row in table.rows
    )
    return _materialize(
        symbols,
        root_family_id=table.root_family_id,
        split=table.split,
        seed=seed,
        master_seed=bundle.provenance.master_seed,
        policy_version=table.policy_version,
    )


def canonical_marker_payload(table: MarkerTableV1) -> str:
    """Alpha-invariant canonical key: roles + typed facts, surfaces erased.

    Rows are canonicalized by content (not ordinal), so renaming every
    surface and permuting row order leaves this digest unchanged.
    """
    payload = sorted(
        (
            row.role,
            row.semantic_type or "",
            row.semantic_role or "",
            row.scope or "",
            row.signature or "",
        )
        for row in table.rows
    )
    return content_sha(payload)


def audit_marker_table(
    bundle: MarkerTableBundleV1,
    *,
    used_surfaces: Iterable[str] = (),
    reference: MarkerTableV1 | None = None,
) -> tuple[MarkerAuditFindingV1, ...]:
    """Run the marker audits. Findings are blocking; see require_clean()."""
    table = bundle.table
    findings: list[MarkerAuditFindingV1] = []
    declared = set(table.surfaces())
    logicals = {entry.logical_surface for entry in bundle.provenance.entries}
    for row in table.rows:
        if row.surface in logicals:
            findings.append(
                MarkerAuditFindingV1(
                    MarkerAuditCode.EXACT_COPY,
                    row.surface,
                    "model-facing surface equals its logical surface",
                )
            )
        for logical in logicals:
            if logical and logical in row.surface and row.surface not in logicals:
                findings.append(
                    MarkerAuditFindingV1(
                        MarkerAuditCode.SEMANTIC_SURFACE,
                        row.surface,
                        f"marker surface embeds logical name {logical!r}",
                    )
                )
    seen: set[str] = set()
    for surface in used_surfaces:
        if surface in seen:
            findings.append(
                MarkerAuditFindingV1(
                    MarkerAuditCode.DUPLICATE, surface, "marker used more than once"
                )
            )
            continue
        seen.add(surface)
        if surface not in declared:
            code = (
                MarkerAuditCode.OUT_OF_RANGE
                if _looks_like_marker(surface)
                else MarkerAuditCode.UNKNOWN
            )
            findings.append(
                MarkerAuditFindingV1(
                    code, surface, "marker used but not declared in the table"
                )
            )
    if reference is not None and (
        canonical_marker_payload(reference) != canonical_marker_payload(table)
    ):
        findings.append(
            MarkerAuditFindingV1(
                MarkerAuditCode.ALPHA_MISMATCH,
                "",
                "canonical AST identity changed under alpha-renaming",
            )
        )
    findings.extend(_ordinal_role_leak_findings(bundle))
    unique = {
        (item.code.value, item.surface, item.evidence): item for item in findings
    }
    return tuple(unique[item] for item in sorted(unique))


def require_clean(
    bundle: MarkerTableBundleV1,
    *,
    used_surfaces: Iterable[str] = (),
    reference: MarkerTableV1 | None = None,
) -> None:
    """Fail closed: raise on any audit finding (publication stop rule)."""
    findings = audit_marker_table(
        bundle, used_surfaces=used_surfaces, reference=reference
    )
    if findings:
        summary = "; ".join(
            f"{item.code.value}:{item.surface or '-'}" for item in findings
        )
        raise ValueError(f"marker audit failed closed: {summary}")


def resolve_marker(table: MarkerTableV1, surface: str) -> RuntimeSymbol:
    """Fail-closed decode-side lookup of a used marker surface."""
    matches = [row for row in table.rows if row.surface == surface]
    if not matches:
        code = (
            MarkerAuditCode.OUT_OF_RANGE
            if _looks_like_marker(surface)
            else MarkerAuditCode.UNKNOWN
        )
        raise ValueError(f"{code.value}: marker {surface!r} is not declared")
    if len(matches) > 1:  # unreachable post-validation; defense in depth
        raise ValueError(f"{MarkerAuditCode.DUPLICATE.value}: {surface!r}")
    return matches[0].to_runtime_symbol()


def assert_role_not_ordinal_recoverable(bundle: MarkerTableBundleV1) -> None:
    """Held-out fixture guard: ordinal alone must not identify a role.

    With per-role surface codecs the prefix is declared authority, so the
    check targets the ordinal stream: ordinal assignment must be deranged
    relative to the logical declaration order recorded in provenance (no
    positional leakage of which logical symbol sits at which ordinal).
    """
    findings = _ordinal_role_leak_findings(bundle)
    if findings:
        summary = "; ".join(item.evidence for item in findings)
        raise ValueError(f"{MarkerAuditCode.ORDINAL_ROLE_LEAK.value}: {summary}")


def _materialize(
    symbols: tuple[RuntimeSymbol, ...],
    *,
    root_family_id: str,
    split: str,
    seed: int,
    master_seed: int,
    policy_version: str,
) -> MarkerTableBundleV1:
    rng = random.Random(seed)
    ordinals = _deranged(range(len(symbols)), rng)
    namespace = hashlib.sha256(f"ns|{seed}".encode()).hexdigest()[:6]
    rows: list[MarkerRowV1] = []
    entries: list[MarkerProvenanceEntryV1] = []
    for index, (symbol, ordinal) in enumerate(zip(symbols, ordinals)):
        role = _RUNTIME_TO_MARKER_ROLE[symbol.role]
        surface = _opaque_surface(role, ordinal, namespace)
        rows.append(
            MarkerRowV1(
                surface=surface,
                role=role,  # type: ignore[arg-type]
                ordinal=ordinal,
                **{
                    key: getattr(symbol, key) for key in _INFERENCE_VISIBLE_FIELDS
                },
            )
        )
        entries.append(
            MarkerProvenanceEntryV1(
                logical_surface=symbol.surface, marker_surface=surface
            )
        )
    rows.sort(key=lambda row: row.ordinal)
    permutation_digest = content_sha(
        {"seed_namespace": namespace, "ordinals": ordinals}
    )
    table_id = content_sha(
        {
            "schema": SCHEMA_VERSION,
            "root_family_id": root_family_id,
            "split": split,
            "policy": policy_version,
            "rows": [row.to_dict() for row in rows],
        }
    )
    table = MarkerTableV1(
        table_id=table_id,
        root_family_id=root_family_id,
        split=split,
        policy_version=policy_version,
        permutation_digest=permutation_digest,
        rows=tuple(rows),
    )
    provenance = MarkerTableProvenanceV1(
        table_id=table_id,
        entries=tuple(entries),
        master_seed=master_seed,
    )
    return MarkerTableBundleV1(table=table, provenance=provenance)


def _symbol_for_row(row: MarkerRowV1, logical_surface: str) -> RuntimeSymbol:
    return RuntimeSymbol(
        surface=logical_surface,
        role=_MARKER_TO_RUNTIME_ROLE[row.role],  # type: ignore[arg-type]
        semantic_type=row.semantic_type,
        semantic_role=row.semantic_role,
        scope=row.scope,
        signature=row.signature,
    )


def _opaque_surface(role: str, ordinal: int, namespace: str) -> str:
    if role == "external_entity":
        return f":x{namespace}{ordinal}"
    if role == "state_ref":
        return f"$s{namespace}{ordinal}"
    if role == "fresh_binder":
        return f"f_{namespace}_{ordinal}"
    return f"b_{namespace}_{ordinal}"


def _require_surface_codec(surface: str, role: str) -> None:
    if role == "external_entity" and not surface.startswith(":"):
        raise ValueError("external_entity markers must start with ':'")
    if role == "state_ref" and not surface.startswith("$"):
        raise ValueError("state_ref markers must start with '$'")
    if role in {"binder", "fresh_binder"} and surface.startswith((":", "$")):
        raise ValueError(f"{role} markers must not use external/state codecs")


def _looks_like_marker(surface: str) -> bool:
    body = surface.lstrip(":$")
    return body.startswith(("b_", "f_", "x", "s")) and any(
        char.isdigit() for char in body
    )


def _deranged(indices: Iterable[int], rng: random.Random) -> list[int]:
    values = list(indices)
    if len(values) < 2:
        return values
    while True:
        permuted = list(values)
        rng.shuffle(permuted)
        if all(a != b for a, b in zip(values, permuted)):
            return permuted


def _ordinal_role_leak_findings(
    bundle: MarkerTableBundleV1,
) -> list[MarkerAuditFindingV1]:
    """Declaration-order position must never equal the assigned ordinal."""
    table = bundle.table
    if len(table.rows) < 2:
        return []
    ordinal_by_surface = {row.surface: row.ordinal for row in table.rows}
    findings: list[MarkerAuditFindingV1] = []
    for index, entry in enumerate(bundle.provenance.entries):
        ordinal = ordinal_by_surface.get(entry.marker_surface)
        if ordinal == index:
            findings.append(
                MarkerAuditFindingV1(
                    MarkerAuditCode.ORDINAL_ROLE_LEAK,
                    entry.marker_surface,
                    f"declaration-order index {index} maps to identical ordinal",
                )
            )
    return findings


def _require_digest(value: str, label: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{label} must be lowercase SHA-256")


__all__ = [
    "MARKER_ROLES",
    "POLICY_VERSION",
    "PROVENANCE_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "MarkerAuditCode",
    "MarkerAuditFindingV1",
    "MarkerProvenanceEntryV1",
    "MarkerRole",
    "MarkerRowV1",
    "MarkerTableBundleV1",
    "MarkerTableProvenanceV1",
    "MarkerTableV1",
    "assert_role_not_ordinal_recoverable",
    "audit_marker_table",
    "canonical_marker_payload",
    "materialize_marker_table",
    "permute_marker_surfaces",
    "require_clean",
    "resolve_marker",
    "split_permutation_seed",
]
