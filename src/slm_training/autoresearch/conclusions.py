"""Hypothesis-family conclusion criteria for the autotrain climb (WP-4).

RC4 of the harness-evolution architecture review found that "the loop is
engineered never to stop, therefore never to conclude": every hard stop had an
automatic bypass, so no amount of adequately powered null evidence ever became
a durable conclusion. This module encodes the repository goal law — *a
rejected experiment closes an approach, never a goal* (AGENTS.md I14) — as
data:

1. **Family key** — a hypothesis family is the canonicalized sorted set of its
   ``lever_keys`` (lowercased, stripped, underscores mapped to hyphens,
   deduplicated, sorted, joined with ``+``). Keys are restricted to the lever
   taxonomy observed in the committed cross-version evidence ledger
   (``resources/experiments/autotrain_climb/evidence_ledger.v1.json``); a
   record naming levers outside that taxonomy can never close a family
   (fail open — we never conclude about levers the durable record has not
   measured).
2. **Closure** — a family concludes after
   ``conclusion_policy.family_close_after_adequately_powered_failures``
   adequately powered failures (``outcome in {confirm_failed,
   ship_rejected}`` with ``n_seeds >= adequate_power_requires.min_seeds`` and
   a decidable measurement when required). Closure appends a content-addressed
   :class:`ClosedApproachRecord` to ``closed_approaches.v1.json`` —
   append-only, stable order, idempotent.
3. **Reopening** — a closed family reopens (stops binding) when a reopen
   condition from the policy holds: a lever key that was not part of the
   closing evidence, or a changed harness component version.

Policy lives in ``policy.v2.json`` (superset of ``policy.v1.json`` plus the
``conclusion_policy`` block); :func:`load_policy` falls back to v1 with a
single logged notice, reusing :func:`climb_policy.load_climb_policy`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from slm_training.autoresearch.climb_policy import (
    CLIMB_RESOURCE_DIR,
    ClimbPolicy,
    ClimbPolicyError,
    load_climb_policy,
)
from slm_training.autoresearch.evidence_ledger import (
    EVAL_KEY_COMPONENTS,
    load_ledger,
)
from slm_training.lineage.records import canonical_json

__all__ = [
    "CLOSED_APPROACHES_SCHEMA",
    "DEFAULT_CLOSED_APPROACHES_PATH",
    "POLICY_V1_PATH",
    "POLICY_V2_PATH",
    "FAILURE_OUTCOMES",
    "ClosedApproachRecord",
    "ConclusionEvidenceRecord",
    "FamilyLedger",
    "binding_closure",
    "canonicalize_lever_key",
    "conclude",
    "conclusion_policy",
    "current_harness_versions",
    "family_key_for_levers",
    "is_family_closed",
    "known_lever_taxonomy",
    "load_closed_approaches",
    "load_policy",
]

logger = logging.getLogger(__name__)

CLOSED_APPROACHES_SCHEMA = "closed_approaches/v1"
POLICY_V1_PATH = CLIMB_RESOURCE_DIR / "policy.v1.json"
POLICY_V2_PATH = CLIMB_RESOURCE_DIR / "policy.v2.json"
DEFAULT_CLOSED_APPROACHES_PATH = CLIMB_RESOURCE_DIR / "closed_approaches.v1.json"

#: Outcomes that count as an approach failure for conclusion purposes.
FAILURE_OUTCOMES = frozenset({"confirm_failed", "ship_rejected"})

#: Fail-closed defaults used only when a loaded policy lacks the
#: ``conclusion_policy`` block (i.e. the v1 fallback). Mirrors policy.v2.json.
_DEFAULT_CONCLUSION_POLICY: Mapping[str, Any] = {
    "family_close_after_adequately_powered_failures": 3,
    "adequate_power_requires": {"min_seeds": 8, "decidable": True},
    "closed_families_reopen_on": ["new_lever_key", "harness_version_change"],
}

_INVARIANT_RE = re.compile(r"\bI\d{1,2}b?\b")
_fallback_notified: set[str] = set()


# ---------------------------------------------------------------------------
# Policy loading (v2 with single-notice v1 fallback)
# ---------------------------------------------------------------------------


def load_policy(resource_dir: Path | str | None = None) -> ClimbPolicy:
    """Load ``policy.v2.json`` when present, else fall back to ``policy.v1.json``.

    Reuses the cached :func:`load_climb_policy` loader (v2 is a strict
    superset of the v1 schema family, so the same validation applies). The
    fallback emits a single logged notice per resource directory.
    """
    base = Path(resource_dir) if resource_dir else CLIMB_RESOURCE_DIR
    v2 = base / POLICY_V2_PATH.name
    v1 = base / POLICY_V1_PATH.name
    if v2.is_file():
        return load_climb_policy(str(v2))
    key = str(base.resolve())
    if key not in _fallback_notified:
        _fallback_notified.add(key)
        logger.info(
            "conclusion policy: %s absent; falling back to %s "
            "(default conclusion_policy applies)",
            v2.name,
            v1,
        )
    return load_climb_policy(str(v1))


def conclusion_policy(policy: ClimbPolicy | None = None) -> Mapping[str, Any]:
    """The ``conclusion_policy`` block (defaults when the v1 fallback lacks it)."""
    pol = policy or load_policy()
    raw = pol.payload.get("conclusion_policy")
    if isinstance(raw, Mapping):
        merged = dict(_DEFAULT_CONCLUSION_POLICY)
        merged.update(dict(raw))
        return merged
    return dict(_DEFAULT_CONCLUSION_POLICY)


# ---------------------------------------------------------------------------
# Family keys + observed lever taxonomy
# ---------------------------------------------------------------------------


def canonicalize_lever_key(key: str) -> str:
    """Canonical lever-key form: stripped, lowercased, underscores → hyphens."""
    return str(key).strip().lower().replace("_", "-")


def family_key_for_levers(lever_keys: Iterable[str]) -> str:
    """Canonical family key: deduplicated sorted canonical keys joined with ``+``."""
    keys = sorted(
        {canonicalize_lever_key(k) for k in lever_keys if str(k).strip()}
    )
    return "+".join(keys)


def _family_levers(family_key: str) -> frozenset[str]:
    return frozenset(
        canonicalize_lever_key(part)
        for part in str(family_key).split("+")
        if part.strip()
    )


def known_lever_taxonomy(ledger_path: Path | None = None) -> frozenset[str]:
    """Lever taxonomy = canonicalized arm slugs of the committed evidence ledger.

    The durable cross-version record (``evidence_ledger.v1.json``) defines
    which levers have actually been measured; conclusions are restricted to
    that taxonomy so a family can never be closed over levers the record has
    never seen.
    """
    arms = load_ledger(ledger_path)
    return frozenset(canonicalize_lever_key(slug) for slug in arms)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


class ConclusionEvidenceRecord(BaseModel):
    """One evidence-ledger row as consumed by the conclusion criteria."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    record_id: str
    lever_keys: tuple[str, ...] = ()
    outcome: str = ""
    n_seeds: int = 0
    decidable: bool = False
    goal_invariant: str | None = None
    hypothesis_text: str | None = None
    reasons: tuple[str, ...] = ()
    harness_versions: Mapping[str, str] = Field(default_factory=dict)
    run_date: str | None = None

    @property
    def family_key(self) -> str:
        return family_key_for_levers(self.lever_keys)

    @property
    def is_failure(self) -> bool:
        return self.outcome in FAILURE_OUTCOMES


def _coerce_record(raw: Mapping[str, Any] | ConclusionEvidenceRecord) -> (
    ConclusionEvidenceRecord
):
    if isinstance(raw, ConclusionEvidenceRecord):
        return raw
    data = dict(raw)
    record_id = str(
        data.get("record_id") or data.get("id") or data.get("evidence_id") or ""
    )
    outcome = str(data.get("outcome") or data.get("status") or "")
    lever_keys = data.get("lever_keys") or data.get("levers") or ()
    reasons = data.get("reasons") or ()
    return ConclusionEvidenceRecord(
        record_id=record_id,
        lever_keys=tuple(str(k) for k in lever_keys),
        outcome=outcome,
        n_seeds=int(data.get("n_seeds") or 0),
        decidable=bool(data.get("decidable", False)),
        goal_invariant=(
            str(data["goal_invariant"]) if data.get("goal_invariant") else None
        ),
        hypothesis_text=(
            str(data["hypothesis_text"]) if data.get("hypothesis_text") else None
        ),
        reasons=tuple(str(r) for r in reasons),
        harness_versions={
            str(k): str(v)
            for k, v in (data.get("harness_versions") or {}).items()
        },
        run_date=str(data["run_date"]) if data.get("run_date") else None,
    )


class ClosedApproachRecord(BaseModel):
    """A concluded hypothesis family — an approach closed on durable evidence.

    A closed approach **never** closes a goal (AGENTS.md I14: goals are
    non-negotiable; approaches are disposable). ``goal_invariant_served``
    names the goal invariant the approach was serving so the goal visibly
    stays open; ``reopen_conditions`` name exactly what re-legalizes the
    family. Records are content-addressed (``id`` = SHA-256 of the family key
    plus the closing evidence ids) and live in an append-only ledger: entries
    are never rewritten or deleted.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    id: str
    family_key: str
    closed_at_run_date: str
    evidence_record_ids: list[str]
    goal_invariant_served: str = "unknown"
    reopen_conditions: list[str]
    #: Harness component versions in force when the family closed; the
    #: ``harness_version_change`` reopen condition compares against these.
    harness_versions: dict[str, str] = Field(default_factory=dict)


def closed_approach_id(family_key: str, evidence_record_ids: Sequence[str]) -> str:
    """Content-addressed closure id: sha256(family key + closing evidence ids)."""
    body = {
        "schema": CLOSED_APPROACHES_SCHEMA,
        "family_key": str(family_key),
        "evidence_record_ids": sorted(str(x) for x in evidence_record_ids),
    }
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


def current_harness_versions(
    components: Sequence[str] = EVAL_KEY_COMPONENTS,
) -> dict[str, str]:
    """Current versions of the eval-comparability harness components.

    Fail-soft: components missing from the registry (or a broken registry)
    are simply absent from the mapping — conclusions must never crash on
    version lookup.
    """
    versions: dict[str, str] = {}
    try:
        from slm_training.harness_core.versioning import component_version
    except Exception:  # noqa: BLE001 — conclusions must not require the registry
        return versions
    for name in components:
        try:
            versions[name] = str(component_version(name))
        except Exception:  # noqa: BLE001
            continue
    return versions


# ---------------------------------------------------------------------------
# Family ledger
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FamilyLedger:
    """Evidence records grouped by canonical hypothesis family.

    Only records whose lever keys all fall inside ``taxonomy`` (the observed
    lever taxonomy of the committed evidence ledger) participate: we never
    conclude about levers the durable record has not measured.
    """

    taxonomy: frozenset[str]
    families: Mapping[str, tuple[ConclusionEvidenceRecord, ...]]
    excluded: tuple[ConclusionEvidenceRecord, ...] = field(default_factory=tuple)

    @classmethod
    def from_records(
        cls,
        records: Iterable[Mapping[str, Any] | ConclusionEvidenceRecord],
        *,
        taxonomy: frozenset[str] | None = None,
    ) -> "FamilyLedger":
        tax = taxonomy if taxonomy is not None else known_lever_taxonomy()
        families: dict[str, list[ConclusionEvidenceRecord]] = {}
        excluded: list[ConclusionEvidenceRecord] = []
        for raw in records:
            record = _coerce_record(raw)
            if not record.record_id or not record.lever_keys:
                excluded.append(record)
                continue
            levers = _family_levers(record.family_key)
            if not levers or not levers <= tax:
                excluded.append(record)
                continue
            families.setdefault(record.family_key, []).append(record)
        return cls(
            taxonomy=tax,
            families={
                key: tuple(rows) for key, rows in sorted(families.items())
            },
            excluded=tuple(excluded),
        )

    def adequately_powered_failures(
        self,
        family_key: str,
        policy_block: Mapping[str, Any] | None = None,
    ) -> tuple[ConclusionEvidenceRecord, ...]:
        """Failure records in the family meeting ``adequate_power_requires``."""
        block = dict(policy_block or conclusion_policy())
        requires = block.get("adequate_power_requires") or {}
        min_seeds = int(requires.get("min_seeds") or 0)
        need_decidable = bool(requires.get("decidable", True))
        rows = self.families.get(family_key, ())
        return tuple(
            r
            for r in rows
            if r.is_failure
            and r.n_seeds >= min_seeds
            and (r.decidable or not need_decidable)
        )


def _derive_goal_invariant(records: Sequence[ConclusionEvidenceRecord]) -> str:
    """The AGENTS.md ``I*`` invariant these records served, else ``unknown``.

    Explicit ``goal_invariant`` fields win; otherwise ``I<n>`` tokens are
    scanned from hypothesis text and reasons. Exactly one distinct invariant
    is required — ambiguity yields ``unknown`` (a closed approach never closes
    a goal, so we never guess which goal it served).
    """
    found: set[str] = set()
    for record in records:
        if record.goal_invariant:
            found.add(record.goal_invariant.strip())
            continue
        text = " ".join((record.hypothesis_text or "", *record.reasons))
        found.update(_INVARIANT_RE.findall(text))
    if len(found) == 1:
        return next(iter(found))
    return "unknown"


# ---------------------------------------------------------------------------
# Closed-approaches ledger IO (append-only)
# ---------------------------------------------------------------------------


def load_closed_approaches(
    path: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load the ledger payload and its raw record dicts (created-if-missing shape)."""
    target = Path(path) if path else DEFAULT_CLOSED_APPROACHES_PATH
    if not target.is_file():
        return (
            {"schema": CLOSED_APPROACHES_SCHEMA, "records": []},
            [],
        )
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ClimbPolicyError(f"invalid closed-approaches ledger {target}: {exc}")
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != CLOSED_APPROACHES_SCHEMA
    ):
        raise ClimbPolicyError(
            f"{target}: schema must be {CLOSED_APPROACHES_SCHEMA!r}"
        )
    records = payload.get("records")
    if not isinstance(records, list):
        raise ClimbPolicyError(f"{target}: 'records' must be a list")
    return dict(payload), [dict(r) for r in records if isinstance(r, Mapping)]


def _write_closed_approaches(
    payload: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    path: Path,
) -> None:
    """Write the append-only closed-approaches ledger atomically.

    Temp file + ``os.replace`` — a process failure mid-write must never
    truncate or corrupt the existing append-only ledger (which would look
    like closed families silently reopening).
    """
    body = dict(payload)
    body["schema"] = CLOSED_APPROACHES_SCHEMA
    body["records"] = list(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(body, indent=2, sort_keys=True) + "\n"
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def _record_reopened(
    record: Mapping[str, Any],
    *,
    current_lever_keys: Iterable[str] | None,
    harness_versions: Mapping[str, str] | None,
) -> bool:
    """Whether a stored closure record has stopped binding under reopen rules."""
    reopen = [str(x) for x in (record.get("reopen_conditions") or [])]
    if "new_lever_key" in reopen and current_lever_keys is not None:
        closed_levers = _family_levers(str(record.get("family_key") or ""))
        current = {
            canonicalize_lever_key(k)
            for k in current_lever_keys
            if str(k).strip()
        }
        if current - closed_levers:
            return True
    if "harness_version_change" in reopen and harness_versions:
        stored = record.get("harness_versions")
        if isinstance(stored, Mapping):
            for name, version in stored.items():
                current_version = harness_versions.get(str(name))
                if current_version is not None and str(current_version) != str(
                    version
                ):
                    return True
    return False


def binding_closure(
    family_key: str,
    current_lever_keys: Iterable[str] | None = None,
    harness_versions: Mapping[str, str] | None = None,
    *,
    path: Path | None = None,
) -> ClosedApproachRecord | None:
    """The still-binding closure record for ``family_key``, if any.

    A record binds until one of its recorded reopen conditions holds: a
    current lever key outside the closing family (``new_lever_key``), or a
    harness component version differing from the versions stored at close
    time (``harness_version_change``). A record without stored harness
    versions cannot detect version change and stays binding on that axis.
    """
    key = family_key_for_levers(_family_levers(family_key))
    _, records = load_closed_approaches(path)
    for raw in records:
        if str(raw.get("family_key") or "") != key:
            continue
        if _record_reopened(
            raw,
            current_lever_keys=current_lever_keys,
            harness_versions=harness_versions,
        ):
            continue
        try:
            return ClosedApproachRecord.model_validate(raw)
        except Exception:  # noqa: BLE001 — malformed entries never bind
            continue
    return None


def is_family_closed(
    family_key: str,
    current_lever_keys: Iterable[str] | None = None,
    harness_versions: Mapping[str, str] | None = None,
    *,
    path: Path | None = None,
) -> bool:
    """Whether ``family_key`` is closed and no reopen condition is met."""
    return (
        binding_closure(
            family_key,
            current_lever_keys=current_lever_keys,
            harness_versions=harness_versions,
            path=path,
        )
        is not None
    )


# ---------------------------------------------------------------------------
# Conclusion
# ---------------------------------------------------------------------------


def conclude(
    records: Iterable[Mapping[str, Any] | ConclusionEvidenceRecord],
    *,
    path: Path | None = None,
    policy: ClimbPolicy | None = None,
    taxonomy: frozenset[str] | None = None,
    harness_versions: Mapping[str, str] | None = None,
    run_date: str | None = None,
) -> list[ClosedApproachRecord]:
    """Append closure records for families crossing the failure threshold.

    Append-only and idempotent: existing entries are never rewritten or
    deleted; a family whose closure is still binding is not re-closed, and
    re-running on the same records adds nothing. New entries are appended in
    sorted family-key order. Returns the newly appended records.
    """
    target = Path(path) if path else DEFAULT_CLOSED_APPROACHES_PATH
    block = conclusion_policy(policy or load_policy())
    threshold = int(
        block.get("family_close_after_adequately_powered_failures") or 0
    )
    if threshold < 1:
        return []
    reopen_conditions = [
        str(x) for x in (block.get("closed_families_reopen_on") or [])
    ]
    versions = (
        dict(harness_versions)
        if harness_versions is not None
        else current_harness_versions()
    )
    closed_at = run_date or datetime.now(timezone.utc).date().isoformat()

    ledger = FamilyLedger.from_records(records, taxonomy=taxonomy)
    payload, existing = load_closed_approaches(target)
    existing_ids = {str(r.get("id") or "") for r in existing}

    appended: list[ClosedApproachRecord] = []
    for family_key in sorted(ledger.families):
        failures = ledger.adequately_powered_failures(family_key, block)
        if len(failures) < threshold:
            continue
        evidence_ids = sorted({r.record_id for r in failures})
        closure_id = closed_approach_id(family_key, evidence_ids)
        if closure_id in existing_ids:
            continue
        still_binding = any(
            str(r.get("family_key") or "") == family_key
            and not _record_reopened(
                r,
                current_lever_keys=_family_levers(family_key),
                harness_versions=versions,
            )
            for r in existing
        )
        if still_binding:
            continue
        record = ClosedApproachRecord(
            id=closure_id,
            family_key=family_key,
            closed_at_run_date=closed_at,
            evidence_record_ids=evidence_ids,
            goal_invariant_served=_derive_goal_invariant(failures),
            reopen_conditions=list(reopen_conditions),
            harness_versions=versions,
        )
        appended.append(record)
        existing_ids.add(closure_id)

    if appended:
        _write_closed_approaches(
            payload,
            [*existing, *(r.model_dump() for r in appended)],
            target,
        )
    return appended
