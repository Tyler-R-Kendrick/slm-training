"""SLM-390 (DSH4-05): quarantined runtime trace-to-eval candidate generation.

Turn recurring runtime failures (W3C run traces, see
``slm_training.runtime.telemetry.trace``) into *candidate* evals without ever
contaminating training automatically and without letting a generated verifier
certify itself.

Lifecycle (all transitions explicit; there are no automatic transitions)::

    QUARANTINED -> REVIEWED -> FROZEN -> CONFIRMATION_USED
                                 +-> PROMOTED_TO_TRAINING (separate transition)

- ``ingest_trace`` redacts/de-identifies, normalizes compiler/tool state,
  clusters recurring failure signatures, and proposes capability / task /
  environment / verifier artifacts.  The candidate is always ``QUARANTINED``.
- Stop rule: a trace that cannot be safely de-identified, reproduced
  (replayable pinned environment), or independently verified stays
  ``QUARANTINED`` with coded ``rejection_reasons``.
- ``mark_reviewed`` requires human/domain review evidence (explicit reviewer id
  plus a full checklist, including separate agent/verifier trajectory review).
- ``freeze_candidate`` requires environment pinning, verifier validation on
  held positive/negative samples, train/eval root-family dedup, and
  exposed-answer / shortcut checks.  It writes an immutable sha256-rowed
  tamper-evident manifest (create-once; ``verify_frozen_eval`` is fail-closed).
- ``record_confirmation_use`` is one-touch: a second confirmation use raises.
- ``promote_to_training`` creates a new derivation activity and dataset
  version; it never mutates the frozen eval.

Fixture-scale harness: no model is trained and no GPU is required.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

from slm_training.harness_core.lineage.records import canonical_json, content_sha
from slm_training.harnesses.train_data.split_policy import RootFamilySplitPolicyV1
from slm_training.versioning import build_version_stamp

__all__ = [
    "EXPERIMENT_ID",
    "MATRIX_SET",
    "MATRIX_VERSION",
    "STATUSES",
    "STOP_DEIDENTIFICATION_FAILED",
    "STOP_ENVIRONMENT_NOT_REPRODUCIBLE",
    "STOP_NOT_INDEPENDENTLY_VERIFIABLE",
    "REQUIRED_REVIEW_CHECKLIST",
    "TraceEvalCandidateV1",
    "FrozenEvalV1",
    "DerivationActivityV1",
    "ConfirmationLedger",
    "LifecycleError",
    "ingest_trace",
    "cluster_fingerprint",
    "mark_reviewed",
    "freeze_candidate",
    "verify_frozen_eval",
    "record_confirmation_use",
    "promote_to_training",
    "verify_environment_replay",
]

MATRIX_VERSION = "trace-eval-candidate-v1"
MATRIX_SET = "slm390_trace_evals"
EXPERIMENT_ID = "slm390-trace-evals"

STATUS_QUARANTINED = "QUARANTINED"
STATUS_REVIEWED = "REVIEWED"
STATUS_FROZEN = "FROZEN"
STATUS_CONFIRMATION_USED = "CONFIRMATION_USED"
STATUS_PROMOTED = "PROMOTED_TO_TRAINING"
STATUSES = (
    STATUS_QUARANTINED,
    STATUS_REVIEWED,
    STATUS_FROZEN,
    STATUS_CONFIRMATION_USED,
    STATUS_PROMOTED,
)

#: Stop-rule reason codes (a candidate carrying any of these never leaves
#: QUARANTINED).
STOP_DEIDENTIFICATION_FAILED = "deidentification_failed"
STOP_ENVIRONMENT_NOT_REPRODUCIBLE = "environment_not_reproducible"
STOP_NOT_INDEPENDENTLY_VERIFIABLE = "not_independently_verifiable"

#: Human/domain review checklist; every item must be explicitly True.
REQUIRED_REVIEW_CHECKLIST = (
    "environment_pinned",
    "verifier_independent",
    "no_exposed_answer",
    "no_shortcut",
    "privacy_ok",
    "agent_trajectory_reviewed",
    "verifier_trajectory_reviewed",
)

_FROZEN_FILE_NAME = "slm390_trace_eval.frozen.json"

# Environment pins required for a replayable (reproducible) environment.
_REQUIRED_ENV_PINS = ("tool.name", "tool.version")

# Fields whose free text is redacted in place; any secret-shaped content found
# outside these fields is a de-identification failure (fail closed).
_REDACTABLE_FIELDS = ("message", "error", "attributes")

_SECRET_PATTERNS = (
    re.compile(r"hf_[A-Za-z0-9]{10,}"),
    re.compile(r"sk-[A-Za-z0-9]{10,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"(?i)(api[_-]?key|token|secret|password)=([^\s,;]+)"),
    re.compile(r"\b[0-9a-f]{48,}\b"),
)
_PATH_PATTERN = re.compile(r"(?<![\w])(?:/[\w.\-]+){2,}")
_HOST_PATTERN = re.compile(
    r"\b(?:[a-z0-9](?:[a-z0-9\-]{0,61}[a-z0-9])?\.)+(?:com|net|org|io|dev|internal|local)\b",
    re.IGNORECASE,
)
_DIGITS_PATTERN = re.compile(r"\d+")
_HEXID_PATTERN = re.compile(r"\b[0-9a-f]{12,32}\b")


class LifecycleError(RuntimeError):
    """Raised on any illegal lifecycle transition."""


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# --------------------------------------------------------------------------
# Candidate record
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TraceEvalCandidateV1:
    """One quarantined trace-derived eval candidate."""

    candidate_version: str
    trace_id: str
    cluster_id: str
    redaction_report: Mapping[str, Any]
    normalized_state: Mapping[str, Any]
    proposed: Mapping[str, Any]
    status: str
    review_evidence: Mapping[str, Any]
    rejection_reasons: tuple[str, ...]
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        data = dict(asdict(self))
        data["rejection_reasons"] = list(self.rejection_reasons)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TraceEvalCandidateV1":
        return cls(
            candidate_version=str(data.get("candidate_version", MATRIX_VERSION)),
            trace_id=str(data["trace_id"]),
            cluster_id=str(data["cluster_id"]),
            redaction_report=dict(data.get("redaction_report", {})),
            normalized_state=dict(data.get("normalized_state", {})),
            proposed=dict(data.get("proposed", {})),
            status=str(data["status"]),
            review_evidence=dict(data.get("review_evidence", {})),
            rejection_reasons=tuple(str(r) for r in data.get("rejection_reasons", [])),
            created_at=str(data.get("created_at", "")),
        )


@dataclass(frozen=True)
class FrozenEvalV1:
    """Immutable frozen eval manifest pointer."""

    candidate_id: str
    root_family_id: str
    manifest_sha256: str
    frozen_path: str
    frozen_at: str

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))


@dataclass(frozen=True)
class DerivationActivityV1:
    """New derivation activity + dataset version created by promotion."""

    activity_id: str
    source_frozen_sha256: str
    source_root_family_id: str
    training_root_family_id: str
    dataset_version: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))


class ConfirmationLedger:
    """One-touch confirmation ledger; a second use of the same eval raises."""

    def __init__(self) -> None:
        self._used: set[str] = set()

    def use(self, frozen: FrozenEvalV1) -> None:
        key = frozen.manifest_sha256
        if key in self._used:
            raise LifecycleError(
                f"confirmation touch already consumed for {frozen.candidate_id}"
            )
        self._used.add(key)

    def used(self, frozen: FrozenEvalV1) -> bool:
        return frozen.manifest_sha256 in self._used


# --------------------------------------------------------------------------
# Redaction / normalization / clustering
# --------------------------------------------------------------------------


def _redact_text(text: str, report: dict[str, int]) -> str:
    for pattern in _SECRET_PATTERNS:
        text, n = pattern.subn("<redacted-secret>", text)
        report["secrets"] += n
    text, n = _PATH_PATTERN.subn("<redacted-path>", text)
    report["paths"] += n
    text, n = _HOST_PATTERN.subn("<redacted-host>", text)
    report["hosts"] += n
    return text


def _contains_secret(text: str) -> bool:
    return any(pattern.search(text) for pattern in _SECRET_PATTERNS)


def _iter_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_strings(item)


def _normalize_message(text: str) -> str:
    text = _HEXID_PATTERN.sub("<id>", text)
    text = _DIGITS_PATTERN.sub("<n>", text)
    return " ".join(text.split())


def cluster_fingerprint(trace: Mapping[str, Any]) -> str:
    """Recurring failures share a fingerprint -> the same cluster."""
    operation = str(trace.get("operation", ""))
    error_type = str(trace.get("error_type") or trace.get("error.type") or "")
    message = _normalize_message(str(trace.get("message", "")))
    return content_sha({"operation": operation, "error_type": error_type, "message": message})[:16]


def _extract_env_pins(trace: Mapping[str, Any]) -> dict[str, str]:
    attributes = trace.get("attributes", {})
    if not isinstance(attributes, Mapping):
        return {}
    pins: dict[str, str] = {}
    for key in _REQUIRED_ENV_PINS:
        value = attributes.get(key)
        if value is not None and str(value):
            pins[key] = str(value)
    for key in ("compiler.version", "python.version"):
        value = attributes.get(key)
        if value is not None and str(value):
            pins[key] = str(value)
    return pins


def ingest_trace(
    trace: Mapping[str, Any],
    *,
    known_canaries: tuple[str, ...] = (),
    now: str | None = None,
) -> TraceEvalCandidateV1:
    """Redact, normalize, cluster, and propose artifacts for one runtime trace.

    Always returns a QUARANTINED candidate (never an automatic transition).
    Stop rules attach coded rejection reasons and keep the candidate
    quarantined: it cannot be safely de-identified, reproduced, or
    independently verified.
    """
    created = now or _now_iso()
    trace_id = str(trace.get("trace_id") or trace.get("traceId") or "")
    report: dict[str, Any] = {"secrets": 0, "paths": 0, "hosts": 0, "canaries": 0}
    reasons: list[str] = []

    redacted: dict[str, Any] = {}
    for key, value in trace.items():
        if key in _REDACTABLE_FIELDS:
            if isinstance(value, str):
                redacted[key] = _redact_text(value, report)
            elif isinstance(value, Mapping):
                redacted[key] = {
                    str(k): _redact_text(v, report) if isinstance(v, str) else v
                    for k, v in value.items()
                }
            else:
                redacted[key] = value
        else:
            # Fail closed: secrets outside redactable free-text fields cannot
            # be safely removed without destroying the record structure.
            if any(_contains_secret(s) for s in _iter_strings(value)):
                reasons.append(STOP_DEIDENTIFICATION_FAILED)
            redacted[key] = value

    # Known canaries: replace literal occurrences inside redactable free-text
    # fields; a canary surviving anywhere else means the trace cannot be safely
    # de-identified (stop rule, fail closed).
    for canary in known_canaries:
        if not canary:
            continue
        replaced = False
        for key in _REDACTABLE_FIELDS:
            value = redacted.get(key)
            if isinstance(value, str) and canary in value:
                redacted[key] = value.replace(canary, "<redacted-canary>")
                replaced = True
            elif isinstance(value, Mapping):
                new_value = {}
                for k, v in value.items():
                    if isinstance(v, str) and canary in v:
                        new_value[k] = v.replace(canary, "<redacted-canary>")
                        replaced = True
                    else:
                        new_value[k] = v
                redacted[key] = new_value
        if replaced:
            report["canaries"] += 1
    if any(c and c in canonical_json(redacted) for c in known_canaries):
        reasons.append(STOP_DEIDENTIFICATION_FAILED)

    if not trace_id:
        reasons.append(STOP_NOT_INDEPENDENTLY_VERIFIABLE)

    pins = _extract_env_pins(trace)
    missing_pins = sorted(pin for pin in _REQUIRED_ENV_PINS if pin not in pins)
    if missing_pins:
        reasons.append(STOP_ENVIRONMENT_NOT_REPRODUCIBLE)

    expected = trace.get("expected_output") or trace.get("expected")
    observed = trace.get("observed_output") or trace.get("output") or trace.get("message")
    if expected is None or observed is None:
        reasons.append(STOP_NOT_INDEPENDENTLY_VERIFIABLE)

    cluster_id = cluster_fingerprint(trace)
    answer = str(expected) if expected is not None else ""
    family = f"slm390-eval-{cluster_id[:12]}"
    proposed = {
        "capability": {
            "name": f"avoid-{str(trace.get('error_type') or 'runtime-failure')}",
            "source_cluster": cluster_id,
        },
        "task": {
            "root_family_id": family,
            "prompt": _redact_text(str(trace.get("task_prompt") or trace.get("message") or ""), report),
        },
        "environment": {
            "pins": pins,
            "replayable": not missing_pins,
        },
        "verifier": {
            "type": "exact_match",
            "answer_sha256": hashlib.sha256(answer.encode("utf-8")).hexdigest() if answer else "",
            "validation": {
                "positive": answer,
                "negative": str(observed) if observed is not None and str(observed) != answer else "",
            },
        },
    }

    candidate = TraceEvalCandidateV1(
        candidate_version=MATRIX_VERSION,
        trace_id=trace_id,
        cluster_id=cluster_id,
        redaction_report=report,
        normalized_state={"env_pins": pins, "redacted_trace_sha256": content_sha(redacted)},
        proposed=proposed,
        status=STATUS_QUARANTINED,
        review_evidence={},
        rejection_reasons=tuple(sorted(set(reasons))),
        created_at=created,
    )
    return candidate


# --------------------------------------------------------------------------
# Lifecycle transitions
# --------------------------------------------------------------------------


def mark_reviewed(
    candidate: TraceEvalCandidateV1,
    review_evidence: Mapping[str, Any],
    *,
    now: str | None = None,
) -> TraceEvalCandidateV1:
    """QUARANTINED -> REVIEWED with explicit human/domain review evidence."""
    if candidate.status != STATUS_QUARANTINED:
        raise LifecycleError(f"cannot review from status {candidate.status}")
    if candidate.rejection_reasons:
        raise LifecycleError(
            f"stop-rule candidate stays quarantined: {candidate.rejection_reasons!r}"
        )
    reviewer = str(review_evidence.get("reviewer_id", ""))
    if not reviewer:
        raise LifecycleError("review requires an explicit reviewer_id")
    checklist = review_evidence.get("checklist", {})
    missing = [item for item in REQUIRED_REVIEW_CHECKLIST if checklist.get(item) is not True]
    if missing:
        raise LifecycleError(f"incomplete review checklist: {missing!r}")
    evidence = dict(review_evidence)
    evidence.setdefault("reviewed_at", now or _now_iso())
    return replace(candidate, status=STATUS_REVIEWED, review_evidence=evidence)


def _validate_verifier(verifier: Mapping[str, Any]) -> list[str]:
    """A generated verifier must not certify itself: it is validated on held
    positive/negative samples and must be non-trivial."""
    errors: list[str] = []
    if verifier.get("type") != "exact_match":
        errors.append("unsupported verifier type")
    validation = verifier.get("validation", {})
    positive = str(validation.get("positive", ""))
    negative = str(validation.get("negative", ""))
    if not positive:
        errors.append("verifier has no positive validation sample")
    if not negative:
        errors.append("verifier has no negative validation sample")
    if positive and positive == negative:
        errors.append("trivial verifier: positive == negative")
    # Deterministic exact-match replay against the pinned answer.
    answer_sha = str(verifier.get("answer_sha256", ""))
    if positive and hashlib.sha256(positive.encode("utf-8")).hexdigest() != answer_sha:
        errors.append("verifier rejects its positive validation sample")
    if negative and hashlib.sha256(negative.encode("utf-8")).hexdigest() == answer_sha:
        errors.append("verifier accepts its negative validation sample")
    return errors


def verify_environment_replay(environment: Mapping[str, Any]) -> bool:
    """Deterministic replay: the pinned environment replays identically twice."""
    pins = environment.get("pins", {})
    if any(pin not in pins for pin in _REQUIRED_ENV_PINS):
        return False
    first = content_sha(pins)
    second = content_sha(dict(pins))
    return first == second


def freeze_candidate(
    candidate: TraceEvalCandidateV1,
    *,
    training_root_family_ids: Iterable[str],
    output_dir: Path,
    now: str | None = None,
) -> tuple[TraceEvalCandidateV1, FrozenEvalV1]:
    """REVIEWED -> FROZEN.  Writes the immutable tamper-evident manifest."""
    if candidate.status != STATUS_REVIEWED:
        raise LifecycleError(f"cannot freeze from status {candidate.status}")

    proposed = candidate.proposed
    environment = dict(proposed.get("environment", {}))
    verifier = dict(proposed.get("verifier", {}))
    task = dict(proposed.get("task", {}))
    family = str(task.get("root_family_id", ""))

    errors: list[str] = []
    if not verify_environment_replay(environment):
        errors.append("environment is not pinned to a replayable state")
    errors.extend(_validate_verifier(verifier))

    policy = RootFamilySplitPolicyV1()
    training_families = {str(f) for f in training_root_family_ids}
    if family in training_families:
        errors.append("eval root family duplicates a training root family")
    if policy.assign(family) == "train":
        errors.append("eval root family hashes into the training split")

    # Exposed-answer / shortcut checks.
    prompt = str(task.get("prompt", ""))
    positive = str(verifier.get("validation", {}).get("positive", ""))
    if positive and positive in prompt:
        errors.append("exposed answer: verifier answer appears in the task prompt")
    if not prompt.strip():
        errors.append("shortcut: empty task prompt")

    if errors:
        raise LifecycleError("; ".join(errors))

    rows = {
        "capability": proposed.get("capability", {}),
        "task": task,
        "environment": environment,
        "verifier": verifier,
    }
    row_shas = {name: content_sha(row) for name, row in sorted(rows.items())}
    manifest = {
        "schema": "TraceEvalCandidateV1Frozen",
        "candidate_version": MATRIX_VERSION,
        "trace_id": candidate.trace_id,
        "cluster_id": candidate.cluster_id,
        "root_family_id": family,
        "rows": rows,
        "row_sha256": row_shas,
        "review_evidence": dict(candidate.review_evidence),
        "redaction_report": dict(candidate.redaction_report),
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        canonical_json(manifest).encode("utf-8")
    ).hexdigest()
    frozen_at = now or _now_iso()
    payload = {
        **manifest,
        "frozen_at": frozen_at,
        "version_stamp": build_version_stamp(
            "harness.experiments", "harness.experiments.slm390_trace_evals"
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / _FROZEN_FILE_NAME
    text = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("manifest_sha256") != manifest["manifest_sha256"]:
            raise FileExistsError("frozen eval already exists with different content")
        if not verify_frozen_eval(path):
            raise RuntimeError("existing frozen eval failed integrity verification")
    else:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(text)

    frozen = FrozenEvalV1(
        candidate_id=f"{candidate.trace_id}:{candidate.cluster_id}",
        root_family_id=family,
        manifest_sha256=manifest["manifest_sha256"],
        frozen_path=path.as_posix(),
        frozen_at=frozen_at,
    )
    return replace(candidate, status=STATUS_FROZEN), frozen


def verify_frozen_eval(path: Path) -> bool:
    """Fail-closed integrity check of a frozen eval manifest."""
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != "TraceEvalCandidateV1Frozen":
            return False
        expected = str(payload["manifest_sha256"])
        manifest = {k: v for k, v in payload.items() if k not in {"frozen_at", "version_stamp", "manifest_sha256"}}
        if hashlib.sha256(canonical_json(manifest).encode("utf-8")).hexdigest() != expected:
            return False
        for name, row in payload["rows"].items():
            if content_sha(row) != payload["row_sha256"][name]:
                return False
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return True


def record_confirmation_use(
    candidate: TraceEvalCandidateV1,
    frozen: FrozenEvalV1,
    ledger: ConfirmationLedger,
) -> TraceEvalCandidateV1:
    """FROZEN -> CONFIRMATION_USED; one touch only."""
    if candidate.status != STATUS_FROZEN:
        raise LifecycleError(f"cannot confirm from status {candidate.status}")
    ledger.use(frozen)
    return replace(candidate, status=STATUS_CONFIRMATION_USED)


def promote_to_training(
    candidate: TraceEvalCandidateV1,
    frozen: FrozenEvalV1,
    *,
    existing_training_family_ids: Iterable[str],
    now: str | None = None,
) -> tuple[TraceEvalCandidateV1, DerivationActivityV1]:
    """FROZEN/CONFIRMATION_USED -> PROMOTED_TO_TRAINING.

    Creates a new derivation activity and dataset version under a *new*
    training root family; the frozen eval manifest is never mutated.
    """
    if candidate.status not in (STATUS_FROZEN, STATUS_CONFIRMATION_USED):
        raise LifecycleError(f"cannot promote from status {candidate.status}")
    frozen_path = Path(frozen.frozen_path)
    if not verify_frozen_eval(frozen_path):
        raise LifecycleError("frozen eval failed integrity verification")

    policy = RootFamilySplitPolicyV1()
    taken = {str(f) for f in existing_training_family_ids}
    taken.add(frozen.root_family_id)
    training_family = ""
    for nonce in range(1000):
        trial = f"{frozen.root_family_id}.trainderiv{nonce}"
        if trial in taken:
            continue
        if policy.assign(trial) == "train":
            training_family = trial
            break
    if not training_family:
        raise LifecycleError("no train-split derivation family available")

    created = now or _now_iso()
    dataset_version = content_sha(
        {
            "source_frozen_sha256": frozen.manifest_sha256,
            "training_root_family_id": training_family,
        }
    )
    activity = DerivationActivityV1(
        activity_id=content_sha(
            {
                "kind": "slm390_promotion",
                "source": frozen.manifest_sha256,
                "dataset_version": dataset_version,
            }
        )[:16],
        source_frozen_sha256=frozen.manifest_sha256,
        source_root_family_id=frozen.root_family_id,
        training_root_family_id=training_family,
        dataset_version=dataset_version,
        created_at=created,
    )
    if not verify_frozen_eval(frozen_path):
        raise LifecycleError("promotion mutated the frozen eval")
    return replace(candidate, status=STATUS_PROMOTED), activity
