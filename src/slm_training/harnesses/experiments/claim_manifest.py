"""SLM-184: single-touch confirmation firewall and preregistered claim manifests.

Wiring/fixture-only harness.  Provides data structures and an access broker that
enforce a preregistered experiment manifest: one confirmation touch per
confirmation suite, with all development touches logged but never counted as
confirmatory evidence.  No model is trained and no GPU is required.
"""

from __future__ import annotations

import json
import hashlib
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from slm_training.versioning import build_version_stamp
from slm_training.lineage.records import canonical_json

__all__ = [
    "EXPERIMENT_ID",
    "MATRIX_SET",
    "MATRIX_VERSION",
    "ExperimentClaimManifestV1",
    "TouchRecord",
    "TouchLedger",
    "SuiteAccessBroker",
    "AccessDecision",
    "validate_manifest",
    "freeze_manifest",
    "is_frozen",
    "classify_iter_artifact",
    "with_claim_manifest_guard",
    "build_default_manifest",
]

MATRIX_VERSION = "claim-manifest-v1"
MATRIX_SET = "slm184_claim_manifest"
EXPERIMENT_ID = "slm184-claim-manifest"

_FROZEN_FILE_NAME = "claim_manifest.frozen.json"
_DEFAULT_CONFIRMATION_TOUCH_LIMIT = 1


@dataclass(frozen=True)
class ExperimentClaimManifestV1:
    """Preregistered claim manifest for a single experiment family."""

    manifest_version: str
    experiment_family_id: str
    source_commit: str
    source_dirty: bool | None
    primary_hypothesis: str
    primary_contrast: str
    primary_endpoint: str
    secondary_endpoints: tuple[str, ...]
    mde: float
    alpha: float
    power: float
    multiplicity_family: str
    allowed_dev_suite_ids: tuple[str, ...]
    confirmation_suite_id: str
    confirmation_suite_digest: str
    confirmation_touch_id: str
    confirmation_touch_limit: int
    frozen_fields: tuple[str, ...]
    tunable_fields: tuple[str, ...]
    selection_rule: str
    stop_rule: str
    seeds: tuple[int, ...]
    hardware_class: str
    checkpoint_pin: str
    config_pin: str
    codec_pin: str
    metric_pin: str
    created_at: str
    author: str

    def to_dict(self) -> dict[str, Any]:
        data = dict(asdict(self))
        data["secondary_endpoints"] = list(self.secondary_endpoints)
        data["allowed_dev_suite_ids"] = list(self.allowed_dev_suite_ids)
        data["frozen_fields"] = list(self.frozen_fields)
        data["tunable_fields"] = list(self.tunable_fields)
        data["seeds"] = list(self.seeds)
        data["source_dirty"] = self.source_dirty
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentClaimManifestV1":
        return cls(
            manifest_version=str(data.get("manifest_version", MATRIX_VERSION)),
            experiment_family_id=str(data["experiment_family_id"]),
            source_commit=str(data.get("source_commit", "")),
            source_dirty=data.get("source_dirty") if "source_dirty" in data else None,
            primary_hypothesis=str(data["primary_hypothesis"]),
            primary_contrast=str(data["primary_contrast"]),
            primary_endpoint=str(data["primary_endpoint"]),
            secondary_endpoints=tuple(str(e) for e in data.get("secondary_endpoints", [])),
            mde=float(data["mde"]),
            alpha=float(data["alpha"]),
            power=float(data["power"]),
            multiplicity_family=str(data["multiplicity_family"]),
            allowed_dev_suite_ids=tuple(str(s) for s in data.get("allowed_dev_suite_ids", [])),
            confirmation_suite_id=str(data["confirmation_suite_id"]),
            confirmation_suite_digest=str(data["confirmation_suite_digest"]),
            confirmation_touch_id=str(data["confirmation_touch_id"]),
            confirmation_touch_limit=int(data.get("confirmation_touch_limit", _DEFAULT_CONFIRMATION_TOUCH_LIMIT)),
            frozen_fields=tuple(str(f) for f in data.get("frozen_fields", [])),
            tunable_fields=tuple(str(f) for f in data.get("tunable_fields", [])),
            selection_rule=str(data["selection_rule"]),
            stop_rule=str(data["stop_rule"]),
            seeds=tuple(int(s) for s in data.get("seeds", [])),
            hardware_class=str(data["hardware_class"]),
            checkpoint_pin=str(data["checkpoint_pin"]),
            config_pin=str(data["config_pin"]),
            codec_pin=str(data["codec_pin"]),
            metric_pin=str(data["metric_pin"]),
            created_at=str(data["created_at"]),
            author=str(data["author"]),
        )


@dataclass(frozen=True)
class TouchRecord:
    """One append-only access touch on a suite."""

    experiment_family_id: str
    suite_digest: str
    touch_id: str
    touch_kind: Literal["dev", "confirm"]
    timestamp: str
    prediction_materialized: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TouchRecord":
        kind = str(data.get("touch_kind", "dev"))
        if kind not in {"dev", "confirm"}:
            kind = "dev"
        return cls(
            experiment_family_id=str(data["experiment_family_id"]),
            suite_digest=str(data["suite_digest"]),
            touch_id=str(data["touch_id"]),
            touch_kind=kind,  # type: ignore[arg-type]
            timestamp=str(data["timestamp"]),
            prediction_materialized=bool(data.get("prediction_materialized", False)),
            reason=str(data.get("reason", "")),
        )


@dataclass(frozen=False)
class TouchLedger:
    """Append-only record of suite touches for an experiment family."""

    records: list[TouchRecord] = field(default_factory=list)

    def record_touch(self, record: TouchRecord) -> None:
        """Append a touch; duplicate touch_ids are rejected."""
        if any(r.touch_id == record.touch_id for r in self.records):
            raise ValueError(f"duplicate touch_id: {record.touch_id}")
        self.records.append(record)

    def confirmation_touches_for(
        self, family_id: str, suite_digest: str
    ) -> list[TouchRecord]:
        return [
            r
            for r in self.records
            if r.experiment_family_id == family_id
            and r.suite_digest == suite_digest
            and r.touch_kind == "confirm"
        ]

    def has_prediction_materialized_touch(
        self, family_id: str, suite_digest: str
    ) -> bool:
        return any(
            r.prediction_materialized
            for r in self.confirmation_touches_for(family_id, suite_digest)
        )

    def to_dict(self) -> dict[str, Any]:
        return {"records": [r.to_dict() for r in self.records]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TouchLedger":
        ledger = cls()
        for row in data.get("records", []):
            ledger.record_touch(TouchRecord.from_dict(row))
        return ledger


@dataclass(frozen=True)
class AccessDecision:
    """Outcome of a suite access request."""

    allowed: bool
    reason: str
    touch_record: TouchRecord | None = None


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _touch_id(manifest: ExperimentClaimManifestV1, suite_id: str, kind: str) -> str:
    return f"{manifest.experiment_family_id}__{suite_id}__{kind}__{_now_iso()}"


class SuiteAccessBroker:
    """Enforce single-touch confirmation access to preregistered suites."""

    def __init__(self, ledger: TouchLedger | None = None) -> None:
        self.ledger = ledger or TouchLedger()

    def request_dev_access(
        self,
        manifest: ExperimentClaimManifestV1,
        suite_id: str,
        suite_digest: str,
        *,
        prediction_materialized: bool = False,
        reason: str = "development exploration",
    ) -> AccessDecision:
        """Log a dev touch and always allow."""
        record = TouchRecord(
            experiment_family_id=manifest.experiment_family_id,
            suite_digest=suite_digest,
            touch_id=_touch_id(manifest, suite_id, "dev"),
            touch_kind="dev",
            timestamp=_now_iso(),
            prediction_materialized=prediction_materialized,
            reason=reason,
        )
        self.ledger.record_touch(record)
        return AccessDecision(
            allowed=True,
            reason="dev access always allowed; touch logged",
            touch_record=record,
        )

    def request_confirmation_access(
        self,
        manifest: ExperimentClaimManifestV1,
        suite_id: str,
        suite_digest: str,
        *,
        frozen_manifest_path: Path | None = None,
        prediction_materialized: bool = False,
        reason: str = "confirmation evaluation",
    ) -> AccessDecision:
        """Allow confirmation access only under the firewall rules."""
        family_id = manifest.experiment_family_id
        errors: list[str] = []

        if frozen_manifest_path is None or not is_frozen(frozen_manifest_path):
            errors.append("manifest is not frozen")
        if suite_id != manifest.confirmation_suite_id:
            errors.append(
                f"suite_id {suite_id!r} does not match confirmation_suite_id "
                f"{manifest.confirmation_suite_id!r}"
            )
        if suite_digest != manifest.confirmation_suite_digest:
            errors.append("suite_digest does not match preregistered digest")
        if self.ledger.has_prediction_materialized_touch(family_id, suite_digest):
            errors.append("a prediction-materialized confirmation touch already exists")

        if errors:
            return AccessDecision(
                allowed=False,
                reason="; ".join(errors),
                touch_record=None,
            )

        record = TouchRecord(
            experiment_family_id=family_id,
            suite_digest=suite_digest,
            touch_id=manifest.confirmation_touch_id,
            touch_kind="confirm",
            timestamp=_now_iso(),
            prediction_materialized=prediction_materialized,
            reason=reason,
        )
        self.ledger.record_touch(record)
        return AccessDecision(
            allowed=True,
            reason="confirmation access granted under preregistered manifest",
            touch_record=record,
        )


def validate_manifest(manifest: ExperimentClaimManifestV1) -> list[str]:
    """Return a list of validation errors; empty list means valid."""
    errors: list[str] = []
    if not manifest.experiment_family_id:
        errors.append("experiment_family_id is required")
    if not manifest.primary_hypothesis:
        errors.append("primary_hypothesis is required")
    if not manifest.primary_contrast:
        errors.append("primary_contrast is required")
    if not manifest.primary_endpoint:
        errors.append("primary_endpoint is required")
    if manifest.mde <= 0.0:
        errors.append("mde must be positive")
    if not 0.0 < manifest.alpha < 1.0:
        errors.append("alpha must be in (0, 1)")
    if not 0.0 < manifest.power < 1.0:
        errors.append("power must be in (0, 1)")
    if not manifest.confirmation_suite_id:
        errors.append("confirmation_suite_id is required")
    if not manifest.confirmation_suite_digest:
        errors.append("confirmation_suite_digest is required")
    if not manifest.confirmation_touch_id:
        errors.append("confirmation_touch_id is required")
    if manifest.confirmation_touch_limit < 1:
        errors.append("confirmation_touch_limit must be >= 1")
    if manifest.confirmation_suite_id in manifest.allowed_dev_suite_ids:
        errors.append(
            "confirmation_suite_id must not be listed in allowed_dev_suite_ids"
        )
    if not manifest.seeds:
        errors.append("seeds must not be empty")
    if not manifest.checkpoint_pin:
        errors.append("checkpoint_pin is required")
    if not manifest.config_pin:
        errors.append("config_pin is required")
    if not manifest.codec_pin:
        errors.append("codec_pin is required")
    if not manifest.metric_pin:
        errors.append("metric_pin is required")
    if not manifest.author:
        errors.append("author is required")
    overlap = set(manifest.frozen_fields) & set(manifest.tunable_fields)
    if overlap:
        errors.append(f"fields cannot be both frozen and tunable: {sorted(overlap)}")
    return errors


def freeze_manifest(
    manifest: ExperimentClaimManifestV1, output_dir: Path
) -> Path:
    """Persist a frozen copy of the manifest and return its path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    frozen_path = output_dir / _FROZEN_FILE_NAME
    manifest_payload = manifest.to_dict()
    manifest_sha = hashlib.sha256(
        canonical_json(manifest_payload).encode("utf-8")
    ).hexdigest()
    if frozen_path.exists():
        if not is_frozen(frozen_path):
            raise RuntimeError("existing frozen manifest failed integrity verification")
        existing = json.loads(frozen_path.read_text(encoding="utf-8"))
        if existing.get("manifest_sha256") != manifest_sha:
            raise FileExistsError(
                "frozen manifest already exists with different content"
            )
        return frozen_path
    payload = {
        "schema": "ExperimentClaimManifestV1Frozen",
        "frozen_at": _now_iso(),
        "manifest": manifest_payload,
        "manifest_sha256": manifest_sha,
        "version_stamp": build_version_stamp(
            "harness.experiments",
            "harness.experiments.claim_manifest",
        ),
    }
    with frozen_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    return frozen_path


def is_frozen(manifest_path: Path) -> bool:
    """Return True only when the create-once frozen payload still matches its digest."""
    if not manifest_path.is_file():
        return False
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = payload["manifest"]
        expected = str(payload["manifest_sha256"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    actual = hashlib.sha256(canonical_json(manifest).encode("utf-8")).hexdigest()
    return (
        payload.get("schema") == "ExperimentClaimManifestV1Frozen"
        and actual == expected
    )


def classify_iter_artifact(data: dict[str, Any]) -> str:
    """Classify a historical iter JSON artifact against the firewall policy."""
    if not isinstance(data, dict):
        return "not_applicable_fixture"

    status = data.get("status", "")
    claim_class = data.get("claim_class", "")
    has_stamp = bool(data.get("version_stamp")) and isinstance(
        data.get("version_stamp"), dict
    )
    has_source_commit = bool(
        data.get("source_commit") or data.get("version_stamp", {}).get("code_commit")
    )
    suite_role = data.get("suite_role", "")
    reused_flags = {
        "eval_from_run",
        "derive_from",
        "difficulty_from",
        "dedup_against",
    }
    has_reuse = any(data.get(k) for k in reused_flags)

    if status in {"plan_only", "development", "dev"} or claim_class in {"development"}:
        return "development_only"

    if has_reuse or data.get("reused_evaluation_data"):
        return "reused_evaluation_data"

    if not has_stamp or not has_source_commit:
        return "provenance_incomplete"

    if (
        status in {"fixture", "confirmatory", "confirmed"}
        and claim_class in {"wiring", "confirmatory"}
        and data.get("confirmation_suite_id")
        and suite_role in {"confirmatory", "confirmation"}
    ):
        return "clean_confirmation"

    if status == "fixture" and claim_class == "wiring":
        return "not_applicable_fixture"

    return "provenance_incomplete"


@contextmanager
def with_claim_manifest_guard(
    manifest_path: Path,
    ledger_path: Path,
    suite_id: str,
    suite_digest: str,
    *,
    is_confirmation: bool = False,
    prediction_materialized: bool = False,
    reason: str = "",
):
    """Context manager that checks suite access before yielding.

    Loads (or creates) the ledger, records the appropriate touch, and fails
    closed when confirmation rules are violated.  Development touches are always
    allowed.  This is a default-off fixture helper; callers must opt in.
    """
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = ExperimentClaimManifestV1.from_dict(
        manifest_data.get("manifest", manifest_data)
    )
    ledger = (
        TouchLedger.from_dict(json.loads(ledger_path.read_text(encoding="utf-8")))
        if ledger_path.is_file()
        else TouchLedger()
    )
    broker = SuiteAccessBroker(ledger)

    default_reason = (
        "confirmation evaluation" if is_confirmation else "development exploration"
    )
    frozen_manifest_path = manifest_path.parent / _FROZEN_FILE_NAME
    if is_confirmation:
        decision = broker.request_confirmation_access(
            manifest,
            suite_id,
            suite_digest,
            frozen_manifest_path=frozen_manifest_path,
            prediction_materialized=prediction_materialized,
            reason=reason or default_reason,
        )
    else:
        decision = broker.request_dev_access(
            manifest,
            suite_id,
            suite_digest,
            prediction_materialized=prediction_materialized,
            reason=reason or default_reason,
        )

    if not decision.allowed:
        raise PermissionError(decision.reason)

    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(
        json.dumps(ledger.to_dict(), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    try:
        yield decision
    finally:
        pass


def build_default_manifest(
    *,
    experiment_family_id: str = "slm184-fixture-family",
    confirmation_suite_id: str = "rico_held",
    confirmation_suite_digest: str = "sha256:0000000000000000000000000000000000000000000000000000000000000000",
) -> ExperimentClaimManifestV1:
    """Build a minimal valid fixture manifest for tests and demonstrations."""
    return ExperimentClaimManifestV1(
        manifest_version=MATRIX_VERSION,
        experiment_family_id=experiment_family_id,
        source_commit="unknown",
        source_dirty=None,
        primary_hypothesis=(
            "A preregistered claim manifest can enforce a single confirmation touch "
            "on one suite while allowing unlimited logged development touches."
        ),
        primary_contrast="unrestricted access vs. single-touch confirmation",
        primary_endpoint="confirmation_access_granted_once_then_denied",
        secondary_endpoints=("ledger_completeness", "digest_mismatch_denial"),
        mde=0.05,
        alpha=0.05,
        power=0.8,
        multiplicity_family="primary_only",
        allowed_dev_suite_ids=("smoke", "held_out", "adversarial", "ood"),
        confirmation_suite_id=confirmation_suite_id,
        confirmation_suite_digest=confirmation_suite_digest,
        confirmation_touch_id=f"{experiment_family_id}__{confirmation_suite_id}__confirm",
        confirmation_touch_limit=_DEFAULT_CONFIRMATION_TOUCH_LIMIT,
        frozen_fields=(
            "experiment_family_id",
            "primary_hypothesis",
            "primary_contrast",
            "primary_endpoint",
            "confirmation_suite_id",
            "confirmation_suite_digest",
            "mde",
            "alpha",
            "power",
        ),
        tunable_fields=("seeds", "hardware_class"),
        selection_rule="single primary endpoint; no cherry-picking across suites",
        stop_rule="stop at first prediction-materialized confirmation touch",
        seeds=(0, 1, 2),
        hardware_class="cpu",
        checkpoint_pin="v1",
        config_pin="v1",
        codec_pin="v1",
        metric_pin="v1",
        created_at=_now_iso(),
        author="slm184-fixture",
    )
