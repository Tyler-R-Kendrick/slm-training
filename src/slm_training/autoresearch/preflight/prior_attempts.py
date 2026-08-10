"""Prior-attempt skeptic: refuse to re-spend on already-refuted configs.

RC3 in ``docs/design/harness-evolution-architecture-review-20260809.md``: the
identical delta 0.3267→0.3828 was rediscovered and rejected in nine separate
loops. This check consults the durable evidence record *before* a screening
cycle is spent:

- ``"block"`` — an identical ``config_fingerprint`` + ``endpoint_metric`` was
  already ``confirm_failed`` / ``ship_rejected`` with adequate power
  (``n_seeds >= 8``, or a meaningful recorded ``p_value``).
- ``"warn"`` — the config was attempted before but only with inadequate power
  (the attempt carries information, not a refutation).
- ``"pass"`` — no matching prior attempt (or a prior *positive*).

Evidence sources, in preference order:

1. ``slm_training.evidence_store.client.find_prior_attempts`` (guarded import;
   the store is being introduced concurrently and may not exist yet).
2. Fallback: the committed arm-level ledger
   ``resources/experiments/autotrain_climb/evidence_ledger.v1.json`` — it has
   no per-config fingerprints, so fallback records match on the candidate's
   optional ``slug`` key and carry ``config_fingerprint=None``.

Fingerprint algorithm (only used when the candidate carries no
``config_fingerprint`` of its own — the continuous driver always passes one
computed by its ``_knobs_fingerprint``): prefer
``slm_training.evidence_store.records.compute_config_fingerprint`` when that
package exists (probed by name, guarded import) so locally computed
fingerprints match store-produced records; otherwise mirror
``_knobs_fingerprint`` from ``scripts/run_autotrain_continuous.py`` — sha256
over canonical JSON (``sort_keys=True``, compact separators, ``default=str``)
of the lever dict with the ``steps`` cycle-jitter key excluded, truncated to
16 hex chars. Match logic itself is algorithm-agnostic string equality, so
records fingerprinted by either producer compare correctly against a
candidate carrying the same producer's hash.

Cross-producer fingerprint convergence: a live candidate's own
``config_fingerprint`` hashes its *concrete lever dict*, while
``sync_evidence_store.py``'s climb-ledger-arm records are fingerprinted over
a *slug-keyed source descriptor* (``evidence_store.records.
compute_arm_fingerprint`` — a deliberately different, cheaper-to-reproduce
domain that needs no lever materialization). These two domains never
coincide by construction, so a candidate carrying a ``slug`` also computes
the arm-fingerprint directly (:func:`arm_fingerprint`) and queries/matches
on it alongside the lever-derived fingerprint — closing the gap without
requiring the store to fingerprint over full lever dicts, and without
weakening ``_knobs_fingerprint``'s established (steps-excluded, 16-hex,
truncated) identity used elsewhere for champion-queue dedup.

Never raises: any internal failure degrades to a ``"warn"`` verdict.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from slm_training.autoresearch.preflight import PreflightVerdict

CHECK_ID = "prior_attempts"

#: Outcomes that count as a completed refutation of the config.
FAILED_OUTCOMES = frozenset({"confirm_failed", "ship_rejected"})

#: Seeds at which a null result is powered enough to close the config.
ADEQUATE_POWER_MIN_SEEDS = 8

#: Keys excluded from the fingerprint (cycle jitter, not config identity) —
#: mirrors ``_FINGERPRINT_EXCLUDE_KEYS`` in ``scripts/run_autotrain_continuous.py``.
FINGERPRINT_EXCLUDE_KEYS = frozenset({"steps"})

_LEDGER_PATH = (
    Path(__file__).resolve().parents[2]
    / "resources"
    / "experiments"
    / "autotrain_climb"
    / "evidence_ledger.v1.json"
)


def config_fingerprint(config: Mapping[str, Any]) -> str:
    """Canonical config identity hash (see module docstring for provenance)."""
    try:  # Prefer the evidence-store helper when that package exists.
        from slm_training.evidence_store import records as _records  # type: ignore

        for name in (
            "compute_config_fingerprint",
            "config_fingerprint",
            "fingerprint_config",
            "fingerprint",
        ):
            helper = getattr(_records, name, None)
            if callable(helper):
                return str(helper(dict(config)))
    except Exception:  # noqa: BLE001 — concurrent module may not exist yet
        pass
    stable = {
        key: value
        for key, value in (config or {}).items()
        if key not in FINGERPRINT_EXCLUDE_KEYS
    }
    payload = json.dumps(stable, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _field(record: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` from a mapping-or-object evidence record."""
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


def _meaningful_p_value(value: Any) -> bool:
    """A recorded p-value that reflects a real completed test."""
    try:
        p = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(p) and 0.0 < p <= 1.0


def _adequate_power(record: Any) -> bool:
    n_seeds = _field(record, "n_seeds")
    try:
        if n_seeds is not None and int(n_seeds) >= ADEQUATE_POWER_MIN_SEEDS:
            return True
    except (TypeError, ValueError):
        pass
    return _meaningful_p_value(_field(record, "p_value"))


def _ledger_fallback_records() -> list[dict[str, Any]]:
    """Synthesize evidence-store-shaped records from the committed arm ledger.

    The arm ledger aggregates by slug (no per-config fingerprints), so the
    records carry ``config_fingerprint=None`` plus a ``slug`` field; matching
    against them uses the candidate's optional ``slug`` key.
    """
    try:
        payload = json.loads(_LEDGER_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    arms = payload.get("arms") if isinstance(payload, Mapping) else None
    if not isinstance(arms, Mapping):
        return []
    records: list[dict[str, Any]] = []
    for slug, stats in sorted(arms.items()):
        if not isinstance(stats, Mapping):
            continue
        n_positive = int(stats.get("n_positive") or 0)
        n_null = int(stats.get("n_null") or 0)
        n_complete = int(stats.get("n_complete") or 0)
        if n_complete <= 0:
            continue
        outcome = (
            "confirm_failed" if n_positive == 0 and n_null > 0 else "attempted"
        )
        records.append(
            {
                "slug": str(slug),
                "config_fingerprint": None,
                "endpoint_metric": "smoke.structural_similarity",
                "outcome": outcome,
                "n_seeds": n_complete,
                "p_value": None,
                "source": "evidence_ledger.v1.json",
            }
        )
    return records


def arm_fingerprint(slug: str) -> str | None:
    """The evidence store's climb-ledger-arm fingerprint for ``slug``, if importable.

    Fingerprint-convergence fix: ``sync_evidence_store.py`` fingerprints
    climb-ledger-arm records by a slug-keyed source descriptor (see
    ``evidence_store.records.compute_arm_fingerprint``), not by the arm's
    concrete lever dict — a *different domain* from the live candidate's own
    ``config_fingerprint`` (lever-derived). Recomputing the same
    slug-keyed fingerprint here lets an exact lookup find those records
    directly, instead of relying only on fuzzy hypothesis/lever-key search.
    """
    try:
        from slm_training.evidence_store.records import compute_arm_fingerprint

        return compute_arm_fingerprint(slug)
    except Exception:  # noqa: BLE001 — store is concurrent work, may not exist
        return None


def _find_prior_attempts(candidate: Mapping[str, Any]) -> tuple[list[Any], str]:
    """Fetch prior-attempt records; returns ``(records, source)``."""
    try:
        from slm_training.evidence_store import client as _client  # type: ignore

        records = list(
            _client.find_prior_attempts(
                hypothesis_text=candidate.get("hypothesis_text"),
                lever_keys=candidate.get("lever_keys"),
                config_fingerprint=candidate.get("config_fingerprint"),
            )
            or []
        )
        slug = candidate.get("slug")
        arm_fp = arm_fingerprint(str(slug)) if slug else None
        if arm_fp and arm_fp != candidate.get("config_fingerprint"):
            seen = {getattr(r, "dedup_key", lambda: id(r))() for r in records}
            for record in _client.find_prior_attempts(config_fingerprint=arm_fp) or []:
                key = getattr(record, "dedup_key", lambda: id(record))()
                if key not in seen:
                    seen.add(key)
                    records.append(record)
        return records, "evidence_store"
    except Exception:  # noqa: BLE001 — store is concurrent work, may not exist
        return _ledger_fallback_records(), "evidence_ledger_fallback"


def _record_summary(record: Any) -> dict[str, Any]:
    return {
        "config_fingerprint": _field(record, "config_fingerprint"),
        "slug": _field(record, "slug"),
        "endpoint_metric": _field(record, "endpoint_metric"),
        "outcome": _field(record, "outcome"),
        "n_seeds": _field(record, "n_seeds"),
        "p_value": _field(record, "p_value"),
    }


class PriorAttemptsCheck:
    """Blocks re-running configs the record already refuted with power."""

    check_id = CHECK_ID

    def run(self, candidate: dict) -> PreflightVerdict:
        try:
            return self._run(candidate)
        except Exception as exc:  # noqa: BLE001 — never raise from a preflight
            return PreflightVerdict(
                check_id=self.check_id,
                verdict="warn",
                reasons=[f"prior_attempts crashed: {type(exc).__name__}: {exc}"],
            )

    def _run(self, candidate: Mapping[str, Any]) -> PreflightVerdict:
        fingerprint = candidate.get("config_fingerprint")
        if not fingerprint:
            levers = candidate.get("levers") or candidate.get("config")
            if isinstance(levers, Mapping) and levers:
                fingerprint = config_fingerprint(levers)
        slug = candidate.get("slug")
        endpoint = candidate.get("endpoint_metric")
        if not fingerprint and not slug:
            return PreflightVerdict(
                check_id=self.check_id,
                verdict="pass",
                reasons=["no config identity (fingerprint or slug) to match"],
                data={"matched": 0},
            )

        records, source = _find_prior_attempts(
            {**dict(candidate), "config_fingerprint": fingerprint}
        )
        arm_fp = arm_fingerprint(str(slug)) if slug else None
        matches: list[Any] = []
        for record in records:
            record_fp = _field(record, "config_fingerprint")
            record_slug = _field(record, "slug")
            if not record_slug:
                # Evidence-store EvidenceRecordV1 rows carry no ``slug``
                # attribute; a single-entry lever_keys list is the arm slug
                # for climb-ledger-arm records (see records.compute_arm_fingerprint).
                record_lever_keys = _field(record, "lever_keys") or []
                if (
                    isinstance(record_lever_keys, (list, tuple))
                    and len(record_lever_keys) == 1
                ):
                    record_slug = record_lever_keys[0]
            if fingerprint and record_fp and str(record_fp) == str(fingerprint):
                matches.append(record)
            elif arm_fp and record_fp and str(record_fp) == str(arm_fp):
                matches.append(record)
            elif slug and record_slug and str(record_slug) == str(slug):
                matches.append(record)

        def endpoint_matches(record: Any) -> bool:
            record_endpoint = _field(record, "endpoint_metric")
            return bool(endpoint) and bool(record_endpoint) and (
                str(record_endpoint) == str(endpoint)
            )

        powered_refutations = [
            record
            for record in matches
            if str(_field(record, "outcome")) in FAILED_OUTCOMES
            and endpoint_matches(record)
            and _adequate_power(record)
        ]
        if powered_refutations:
            return PreflightVerdict(
                check_id=self.check_id,
                verdict="block",
                reasons=[
                    "identical config + endpoint already refuted with adequate "
                    f"power ({len(powered_refutations)} record(s), source={source})",
                    *[
                        json.dumps(_record_summary(record), sort_keys=True, default=str)
                        for record in powered_refutations[:3]
                    ],
                ],
                data={
                    "source": source,
                    "matched": len(matches),
                    "powered_refutations": len(powered_refutations),
                    "config_fingerprint": fingerprint,
                },
            )
        if matches:
            return PreflightVerdict(
                check_id=self.check_id,
                verdict="warn",
                reasons=[
                    f"config attempted before without adequate power "
                    f"({len(matches)} record(s), source={source}); a repeat must "
                    "add power (seeds), not merely re-observe",
                ],
                data={
                    "source": source,
                    "matched": len(matches),
                    "powered_refutations": 0,
                    "config_fingerprint": fingerprint,
                },
            )
        return PreflightVerdict(
            check_id=self.check_id,
            verdict="pass",
            reasons=[f"no matching prior attempt (source={source})"],
            data={"source": source, "matched": 0, "config_fingerprint": fingerprint},
        )


CHECK = PriorAttemptsCheck()
