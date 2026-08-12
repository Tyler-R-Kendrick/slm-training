"""RESEARCH-02 default-off experiment preregistration registry (SLM-533).

Machine-readable owner:
``src/slm_training/resources/research_experiment_preregistry.json``.

Human-readable owner: ``docs/design/research-experiment-preregistry.md``.

Uses RESEARCH-01 citation grounding and ``ExperimentCampaignV1`` / ``CampaignLockV1``
for execution locks. Filing or compiling an entry is not evidence.
"""

from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from slm_training.citation_catalog import require_grounded_citations
from slm_training.harness_core.lineage.records import canonical_json

SCHEMA = "research_experiment_preregistry/v1"
REGISTRY_RELATIVE = "src/slm_training/resources/research_experiment_preregistry.json"
ISSUE_KEY = "RESEARCH-02"
LINEAR_ID = "SLM-533"
DESIGN_DOC = "docs/design/research-experiment-preregistry.md"

DISPOSITIONS = frozenset(
    {
        "registered",
        "blocked",
        "executable",
        "completed",
        "rejected",
        "reopened",
    }
)
TERMINAL_DISPOSITIONS = frozenset({"completed", "rejected"})
BLOCKER_STATUSES = frozenset({"open", "completed", "waived"})
ACTIVATION_PREFLIGHT_REQUIRED = frozenset(
    {"blockers_complete", "campaign_lock", "citations_grounded", "default_off"}
)

REQUIRED_ENTRY_KEYS = frozenset(
    {
        "experiment_key",
        "linear_id",
        "hypothesis",
        "cited_lineage",
        "repo_motivation",
        "blockers",
        "matched_control",
        "primary_metric",
        "secondary_safety_metrics",
        "decision_rule",
        "activation_preflight",
        "resource_cap",
        "falsifier",
        "evidence_path",
        "promotion_boundary",
        "knobs",
        "disposition",
        "default_off",
        "research_only",
        "knob_hypothesis_signature",
        "campaign_lock_sha256",
    }
)

_EXPERIMENT_KEY_RE = re.compile(r"^RESEARCH-(?:0[3-9]|1\d|20)$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_LINEAR_RE = re.compile(r"^SLM-\d+$")


class ResearchPreregistryError(ValueError):
    """Fail-closed preregistration / activation rejection."""


def registry_path(*, root: Path | None = None) -> Path:
    base = root if root is not None else Path(__file__).resolve().parents[2]
    return base / REGISTRY_RELATIVE


@lru_cache(maxsize=4)
def _load_raw(path_str: str) -> dict[str, Any]:
    return json.loads(Path(path_str).read_text(encoding="utf-8"))


def load_registry(*, root: Path | None = None) -> dict[str, Any]:
    path = registry_path(root=root)
    doc = _load_raw(str(path))
    if doc.get("schema") != SCHEMA:
        raise ResearchPreregistryError(
            f"unexpected registry schema {doc.get('schema')!r}; expected {SCHEMA!r}"
        )
    return doc


def iter_experiments(*, root: Path | None = None) -> list[dict[str, Any]]:
    return list(load_registry(root=root)["experiments"])


def experiment_by_key(
    experiment_key: str, *, root: Path | None = None
) -> dict[str, Any]:
    for row in iter_experiments(root=root):
        if row["experiment_key"] == experiment_key:
            return row
    raise KeyError(experiment_key)


def normalize_hypothesis(hypothesis: str) -> str:
    return " ".join(str(hypothesis).split()).strip().lower()


def compute_knob_hypothesis_signature(
    *, knobs: Mapping[str, Any], hypothesis: str
) -> str:
    """Stable signature over knobs + normalized hypothesis."""
    payload = {
        "hypothesis": normalize_hypothesis(hypothesis),
        "knobs": {k: knobs[k] for k in sorted(knobs) if knobs[k] is not None},
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _require_nonempty_str(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ResearchPreregistryError(
            f"{row.get('experiment_key')!r}: {key} must be a non-empty string"
        )
    return value.strip()


def validate_entry(
    row: Mapping[str, Any],
    *,
    root: Path | None = None,
    check_citations: bool = True,
) -> str:
    """Validate one registry row; return its signature. Fail closed on gaps."""
    missing = sorted(REQUIRED_ENTRY_KEYS - set(row))
    if missing:
        raise ResearchPreregistryError(f"entry missing keys {missing}")

    key = _require_nonempty_str(row, "experiment_key")
    if not _EXPERIMENT_KEY_RE.match(key):
        raise ResearchPreregistryError(f"invalid experiment_key {key!r}")

    linear = _require_nonempty_str(row, "linear_id")
    if not _LINEAR_RE.match(linear):
        raise ResearchPreregistryError(f"{key}: invalid linear_id {linear!r}")

    hypothesis = _require_nonempty_str(row, "hypothesis")
    if len(hypothesis) < 12:
        raise ResearchPreregistryError(f"{key}: hypothesis too short")

    for field in (
        "repo_motivation",
        "matched_control",
        "primary_metric",
        "decision_rule",
        "falsifier",
        "evidence_path",
        "promotion_boundary",
    ):
        _require_nonempty_str(row, field)

    if row.get("default_off") is not True:
        raise ResearchPreregistryError(f"{key}: default_off must be true")
    if row.get("research_only") is not True:
        raise ResearchPreregistryError(f"{key}: research_only must be true")

    disposition = row.get("disposition")
    if disposition not in DISPOSITIONS:
        raise ResearchPreregistryError(f"{key}: unknown disposition {disposition!r}")

    citations = row.get("cited_lineage")
    if not isinstance(citations, list) or not citations:
        raise ResearchPreregistryError(f"{key}: cited_lineage required")
    if any(not isinstance(c, str) or not c.strip() for c in citations):
        raise ResearchPreregistryError(f"{key}: cited_lineage entries must be strings")
    if check_citations:
        require_grounded_citations(list(citations), root=root)

    blockers = row.get("blockers")
    if not isinstance(blockers, list) or not blockers:
        raise ResearchPreregistryError(f"{key}: blockers required (fail closed)")
    for blocker in blockers:
        if not isinstance(blocker, Mapping):
            raise ResearchPreregistryError(f"{key}: blocker must be object")
        bid = blocker.get("blocker_id")
        if not isinstance(bid, str) or not bid.strip():
            raise ResearchPreregistryError(f"{key}: blocker_id required")
        status = blocker.get("status")
        if status not in BLOCKER_STATUSES:
            raise ResearchPreregistryError(
                f"{key}: blocker {bid!r} status must be one of {sorted(BLOCKER_STATUSES)}"
            )
        note = blocker.get("note")
        if not isinstance(note, str) or not note.strip():
            raise ResearchPreregistryError(f"{key}: blocker {bid!r} note required")

    safety = row.get("secondary_safety_metrics")
    if not isinstance(safety, list) or not safety:
        raise ResearchPreregistryError(f"{key}: secondary_safety_metrics required")
    if any(not isinstance(m, str) or not m.strip() for m in safety):
        raise ResearchPreregistryError(f"{key}: secondary_safety_metrics must be strings")

    preflight = row.get("activation_preflight")
    if not isinstance(preflight, list) or not preflight:
        raise ResearchPreregistryError(f"{key}: activation_preflight required")
    missing_pf = ACTIVATION_PREFLIGHT_REQUIRED - set(preflight)
    if missing_pf:
        raise ResearchPreregistryError(
            f"{key}: activation_preflight missing {sorted(missing_pf)}"
        )

    cap = row.get("resource_cap")
    if not isinstance(cap, Mapping):
        raise ResearchPreregistryError(f"{key}: resource_cap required")
    for cap_key in ("max_run_minutes", "max_gpu_hours"):
        if cap_key not in cap:
            raise ResearchPreregistryError(f"{key}: resource_cap.{cap_key} required")
        value = cap[cap_key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ResearchPreregistryError(
                f"{key}: resource_cap.{cap_key} must be a non-negative number"
            )
    if not str(cap.get("notes", "")).strip():
        raise ResearchPreregistryError(f"{key}: resource_cap.notes required")

    knobs = row.get("knobs")
    if not isinstance(knobs, Mapping) or not knobs:
        raise ResearchPreregistryError(f"{key}: knobs required for signature")

    expected = compute_knob_hypothesis_signature(knobs=knobs, hypothesis=hypothesis)
    stored = row.get("knob_hypothesis_signature")
    if stored != expected:
        raise ResearchPreregistryError(
            f"{key}: knob_hypothesis_signature mismatch "
            f"(stored={stored!r}, expected={expected!r})"
        )

    lock = row.get("campaign_lock_sha256")
    if lock is not None:
        if not isinstance(lock, str) or not _SHA256_RE.match(lock):
            raise ResearchPreregistryError(
                f"{key}: campaign_lock_sha256 must be 64-hex or null"
            )

    if disposition == "reopened":
        reopen = row.get("reopen_evidence_path")
        if not isinstance(reopen, str) or not reopen.strip():
            raise ResearchPreregistryError(
                f"{key}: reopened disposition requires reopen_evidence_path"
            )

    return expected


def validate_registry_document(
    doc: Mapping[str, Any],
    *,
    root: Path | None = None,
    check_citations: bool = True,
) -> list[str]:
    if doc.get("schema") != SCHEMA:
        raise ResearchPreregistryError(f"schema must be {SCHEMA!r}")
    if doc.get("issue_key") != ISSUE_KEY:
        raise ResearchPreregistryError(f"issue_key must be {ISSUE_KEY!r}")
    if doc.get("linear_id") != LINEAR_ID:
        raise ResearchPreregistryError(f"linear_id must be {LINEAR_ID!r}")
    if doc.get("design_doc") != DESIGN_DOC:
        raise ResearchPreregistryError(f"design_doc must be {DESIGN_DOC!r}")
    if doc.get("verified_by") != "scripts/verify_research_experiment_preregistry.py":
        raise ResearchPreregistryError("verified_by pointer mismatch")
    if doc.get("default_off") is not True or doc.get("research_only") is not True:
        raise ResearchPreregistryError("registry authority must be default_off/research_only")

    experiments = doc.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ResearchPreregistryError("experiments must be a non-empty list")

    required_pilots = doc.get("required_pilot_keys")
    if not isinstance(required_pilots, list) or not required_pilots:
        raise ResearchPreregistryError("required_pilot_keys required")

    seen_keys: set[str] = set()
    sig_owners: dict[str, str] = {}
    for row in experiments:
        if not isinstance(row, Mapping):
            raise ResearchPreregistryError("each experiment must be an object")
        sig = validate_entry(row, root=root, check_citations=check_citations)
        key = row["experiment_key"]
        if key in seen_keys:
            raise ResearchPreregistryError(f"duplicate experiment_key {key!r}")
        seen_keys.add(key)
        prior = sig_owners.get(sig)
        if prior is not None:
            prior_row = next(r for r in experiments if r["experiment_key"] == prior)
            if prior_row.get("disposition") in TERMINAL_DISPOSITIONS:
                if row.get("disposition") != "reopened":
                    raise ResearchPreregistryError(
                        f"signature {sig[:12]}… owned by terminal "
                        f"{prior}/{prior_row['disposition']}; reject equivalent "
                        f"{key} unless disposition=reopened with new evidence"
                    )
            elif row.get("disposition") in TERMINAL_DISPOSITIONS:
                pass  # current is terminal; prior non-terminal still blocks duplicate live
            else:
                raise ResearchPreregistryError(
                    f"duplicate knob/hypothesis signature between {prior!r} and {key!r}"
                )
        else:
            # Also block registering a live equivalent of a terminal elsewhere.
            for other in experiments:
                if other is row:
                    continue
                if other.get("knob_hypothesis_signature") != sig:
                    continue
                if other.get("disposition") in TERMINAL_DISPOSITIONS and row.get(
                    "disposition"
                ) not in {"reopened", *TERMINAL_DISPOSITIONS}:
                    raise ResearchPreregistryError(
                        f"{key}: equivalent of terminal {other['experiment_key']} "
                        f"({other['disposition']}); reopen with evidence or change knobs"
                    )
        sig_owners[sig] = key

    missing_pilots = sorted(set(required_pilots) - seen_keys)
    if missing_pilots:
        raise ResearchPreregistryError(
            f"required RESEARCH pilots missing from registry: {missing_pilots}"
        )
    return sorted(seen_keys)


def blockers_complete(row: Mapping[str, Any]) -> bool:
    blockers = row.get("blockers") or []
    return all(
        isinstance(b, Mapping) and b.get("status") in {"completed", "waived"}
        for b in blockers
    )


def assert_execution_allowed(row: Mapping[str, Any]) -> None:
    """Fail closed unless blockers done, campaign locked, and authority holds.

    Filing / compiling an experiment is never evidence and never grants execution.
    """
    key = row.get("experiment_key", "<unknown>")
    if row.get("default_off") is not True:
        raise ResearchPreregistryError(f"{key}: execution refused (default_off!=true)")
    if row.get("research_only") is not True:
        raise ResearchPreregistryError(f"{key}: execution refused (research_only!=true)")
    if not blockers_complete(row):
        raise ResearchPreregistryError(
            f"{key}: execution refused — blockers incomplete"
        )
    lock = row.get("campaign_lock_sha256")
    if not isinstance(lock, str) or not _SHA256_RE.match(lock):
        raise ResearchPreregistryError(
            f"{key}: execution refused — campaign_lock_sha256 required "
            "(exact ExperimentCampaignV1 / CampaignLockV1 digest)"
        )
    if row.get("disposition") in TERMINAL_DISPOSITIONS:
        raise ResearchPreregistryError(
            f"{key}: execution refused — disposition {row.get('disposition')!r}"
        )
    if row.get("disposition") not in {"executable", "reopened", "registered", "blocked"}:
        raise ResearchPreregistryError(
            f"{key}: execution refused — disposition {row.get('disposition')!r}"
        )
    # registered/blocked still need lock+blockers; disposition alone is not enough.
    if row.get("disposition") == "blocked":
        raise ResearchPreregistryError(f"{key}: execution refused — disposition=blocked")


def reject_equivalent_rerun(
    *,
    candidate: Mapping[str, Any],
    historical: Sequence[Mapping[str, Any]],
) -> None:
    """Reject candidate when a completed/rejected equivalent exists without reopen."""
    validate_entry(candidate, check_citations=False)
    sig = candidate["knob_hypothesis_signature"]
    for prior in historical:
        if prior.get("knob_hypothesis_signature") != sig:
            continue
        if prior.get("disposition") not in TERMINAL_DISPOSITIONS:
            continue
        if candidate.get("disposition") == "reopened" and str(
            candidate.get("reopen_evidence_path", "")
        ).strip():
            continue
        raise ResearchPreregistryError(
            f"reject equivalent of {prior.get('experiment_key')} "
            f"({prior.get('disposition')}): change knobs/hypothesis or reopen "
            "with reopen_evidence_path"
        )
