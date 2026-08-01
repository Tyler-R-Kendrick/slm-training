"""Content-addressed campaign artifacts and append-only telemetry."""

from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from slm_training.autoresearch.schemas import (
    AutotrainActionReceiptV1,
    AutotrainActionV1,
    AutotrainCycleHandoffV1,
    CampaignSpec,
    Diagnosis,
    ExperimentOutcome,
    HypothesisFeedback,
    HypothesisMatrix,
    evaluation_measurement_incomplete,
    utc_now,
)
from slm_training.autoresearch.experiment_campaign import (
    CampaignDeviationV1,
    CampaignLockV1,
    CampaignResultV1,
    ExperimentCampaignV1,
    campaign_manifest_sha256,
    validate_result_claim,
)
from slm_training.lineage.records import canonical_json
from slm_training.levers import MAX_RUN_SECONDS

_OUTCOME_BOUNDARY_EVENTS = frozenset(
    {
        "experiment_started",
        "experiment_finished",
        "outcome_diagnosed",
        "hypothesizer_feedback_recorded",
    }
)
_PREREQUISITE_ACTION_KINDS = frozenset(
    {
        "stop_campaign",
        "repair_harness",
        "repair_formal",
        "rebuild_data",
        "document",
        "deliver_stack",
    }
)
_EXECUTION_ACTION_KINDS = frozenset({"retry_measurement", "next_experiment", "monitor"})


def autotrain_action_sha256(action: AutotrainActionV1) -> str:
    return _sha(action.model_dump(mode="json"))


def append_autotrain_action_receipt(
    root: Path | str, receipt: AutotrainActionReceiptV1
) -> Path:
    path = Path(root) / "loops" / receipt.loop_id / "action_receipts.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(path.with_suffix(".lock"), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(receipt.model_dump_json() + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
    return path


def _pending_autotrain_actions(
    root: Path | str,
    handoff: AutotrainCycleHandoffV1,
    *,
    kinds: frozenset[str],
) -> tuple[tuple[int, AutotrainActionV1], ...]:
    path = Path(root) / "loops" / handoff.loop_id / "action_receipts.jsonl"
    completed: set[tuple[int, str]] = set()
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                receipt = AutotrainActionReceiptV1.model_validate_json(line)
            except ValueError:
                continue
            if (
                receipt.campaign_id == handoff.campaign_id
                and receipt.status == "completed"
            ):
                completed.add((receipt.action_index, receipt.action_sha256))
    return tuple(
        (index, action)
        for index, action in enumerate(handoff.actions)
        if action.kind in kinds
        and (index, autotrain_action_sha256(action)) not in completed
    )


def pending_autotrain_actions(
    root: Path | str, handoff: AutotrainCycleHandoffV1
) -> tuple[tuple[int, AutotrainActionV1], ...]:
    """Return unacknowledged actions that block successor initialization."""

    return _pending_autotrain_actions(root, handoff, kinds=_PREREQUISITE_ACTION_KINDS)


def pending_autotrain_execution_actions(
    root: Path | str, handoff: AutotrainCycleHandoffV1
) -> tuple[tuple[int, AutotrainActionV1], ...]:
    """Return unacknowledged steering actions for the successor cycle."""

    return _pending_autotrain_actions(root, handoff, kinds=_EXECUTION_ACTION_KINDS)


def _payload(value: BaseModel | dict[str, Any]) -> dict[str, Any]:
    return value.model_dump(mode="json") if isinstance(value, BaseModel) else value


def _sha(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


class CampaignStore:
    """One durable local source of truth; remote sinks mirror this directory."""

    EVENT_COLUMNS = (
        "timestamp",
        "event_id",
        "previous_event_sha256",
        "event_type",
        "campaign_id",
        "experiment_id",
        "status",
        "artifact_sha256",
        "detail",
    )

    def __init__(
        self,
        campaign_id: str,
        root: Path | str = Path("outputs/autoresearch"),
    ) -> None:
        self.campaign_id = campaign_id
        self.root = Path(root) / campaign_id

    def initialize(self, campaign: CampaignSpec) -> Path:
        if campaign.campaign_id != self.campaign_id:
            raise ValueError("campaign id does not match store")
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / "campaign.json"
        data = (
            json.dumps(campaign.model_dump(mode="json"), indent=2, sort_keys=True)
            + "\n"
        )
        if path.exists():
            if path.read_text(encoding="utf-8") != data:
                raise FileExistsError(
                    f"campaign already exists with different spec: {path}"
                )
            return path
        self._atomic_new(path, data)
        artifact = self.write_artifact("campaign", campaign)
        self.append_event("campaign_initialized", artifact_sha256=artifact.stem)
        return path

    def load_campaign(self) -> CampaignSpec:
        return CampaignSpec.model_validate_json(
            (self.root / "campaign.json").read_text(encoding="utf-8")
        )

    def lock_experiment_campaign(
        self, manifest: ExperimentCampaignV1
    ) -> CampaignLockV1:
        """Record the one authoritative manifest before execution starts."""
        mutation_lock = self.root / ".campaign-mutation.lock"
        mutation_lock.parent.mkdir(parents=True, exist_ok=True)
        lock_fd = os.open(mutation_lock, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            return self._lock_experiment_campaign(manifest)
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)

    def _lock_experiment_campaign(
        self, manifest: ExperimentCampaignV1
    ) -> CampaignLockV1:
        manifest = ExperimentCampaignV1.model_validate(manifest.model_dump(mode="json"))
        if manifest.campaign_id != self.campaign_id:
            raise ValueError("campaign manifest belongs to a different campaign")
        events = self.verify_event_chain()
        experiment_events = [
            row for row in events if row.get("experiment_id") == manifest.experiment_id
        ]
        digest = campaign_manifest_sha256(manifest)
        locks = [
            row
            for row in experiment_events
            if row.get("event_type") == "experiment_campaign_locked"
        ]
        if locks:
            locked_digest = str(locks[0].get("detail", {}).get("manifest_sha256", ""))
            if locked_digest != digest:
                raise FileExistsError(
                    "experiment campaign is already locked with different content"
                )
            return self.load_experiment_campaign(manifest.experiment_id)
        if any(
            row.get("event_type") in _OUTCOME_BOUNDARY_EVENTS
            for row in experiment_events
        ):
            raise RuntimeError("cannot lock a campaign after experiment outcome access")
        lock = CampaignLockV1(manifest_sha256=digest, manifest=manifest)
        path = self.write_artifact("experiment_campaigns", lock)
        self.append_event(
            "experiment_campaign_locked",
            experiment_id=manifest.experiment_id,
            status="locked",
            artifact_sha256=path.stem,
            detail={"manifest_sha256": digest},
        )
        return lock

    def load_experiment_campaign(self, experiment_id: str) -> CampaignLockV1:
        """Load and verify the earliest pre-start campaign lock."""
        events = self.verify_event_chain()
        first_boundary = next(
            (
                index
                for index, row in enumerate(events)
                if row.get("experiment_id") == experiment_id
                and row.get("event_type") in _OUTCOME_BOUNDARY_EVENTS
            ),
            len(events),
        )
        lock_rows = [
            (index, row)
            for index, row in enumerate(events)
            if row.get("experiment_id") == experiment_id
            and row.get("event_type") == "experiment_campaign_locked"
        ]
        if not lock_rows:
            raise FileNotFoundError(
                f"no locked experiment campaign for {experiment_id}"
            )
        index, row = lock_rows[0]
        if index >= first_boundary:
            raise RuntimeError(
                "campaign lock was not recorded before experiment outcome access"
            )
        if len(lock_rows) != 1:
            raise RuntimeError("multiple campaign locks recorded for experiment")
        artifact_sha = str(row.get("artifact_sha256", ""))
        path = self.root / "artifacts" / "experiment_campaigns" / f"{artifact_sha}.json"
        lock = CampaignLockV1.model_validate_json(path.read_text(encoding="utf-8"))
        event_sha = str(row.get("detail", {}).get("manifest_sha256", ""))
        if lock.manifest_sha256 != event_sha:
            raise RuntimeError("campaign lock event digest mismatch")
        return lock

    def append_campaign_deviation(self, deviation: CampaignDeviationV1) -> Path:
        """Append an exploratory deviation without replacing the locked plan."""
        lock = self.load_experiment_campaign(deviation.experiment_id)
        if deviation.campaign_id != self.campaign_id:
            raise ValueError("deviation belongs to a different campaign")
        if deviation.manifest_sha256 != lock.manifest_sha256:
            raise ValueError("deviation manifest digest does not match lock")
        path = self.write_artifact("campaign_deviations", deviation)
        self.append_event(
            "campaign_deviation_appended",
            experiment_id=deviation.experiment_id,
            status="exploratory",
            artifact_sha256=path.stem,
            detail={"manifest_sha256": lock.manifest_sha256},
        )
        return path

    def validate_campaign_result(
        self,
        result: CampaignResultV1,
        *,
        artifact_root: Path,
        locked_manifest_path: Path | None = None,
    ) -> tuple[str, ...]:
        """Validate a result against the authoritative lock and event history."""
        lock = self.load_experiment_campaign(result.experiment_id)
        failures = list(
            validate_result_claim(
                lock.manifest,
                result,
                artifact_root=artifact_root,
                locked_manifest_path=locked_manifest_path,
                # Default climb enforcement for promotion-class claims; result
                # may carry primary_endpoint_seed_values + EG_params fields.
            )
        )
        events = self.verify_event_chain()
        relevant = [
            row for row in events if row.get("experiment_id") == result.experiment_id
        ]
        if any(
            row.get("event_type") == "campaign_deviation_appended" for row in relevant
        ):
            failures.append("exploratory_deviations_present")
        starts = [
            row for row in relevant if row.get("event_type") == "experiment_started"
        ]
        finishes = [
            row for row in relevant if row.get("event_type") == "experiment_finished"
        ]
        if not starts or any(
            row.get("detail", {}).get("campaign_manifest_sha256")
            != lock.manifest_sha256
            for row in (*starts, *finishes)
        ):
            failures.append("execution_manifest_binding_missing")
        if not finishes:
            failures.append("experiment_finish_missing")
        return tuple(dict.fromkeys(failures))

    def verify_event_chain(self) -> list[dict[str, Any]]:
        """Fail closed on edited, reordered, deleted-link, or forked events."""
        path = self.root / "events.jsonl"
        if not path.exists():
            return []
        events = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        previous = ""
        seen: set[str] = set()
        for event in events:
            event_id = str(event.get("event_id", ""))
            payload = dict(event)
            payload.pop("event_id", None)
            if event_id != _sha(payload):
                raise RuntimeError("event_id digest mismatch")
            if event_id in seen:
                raise RuntimeError("duplicate event_id in campaign chain")
            if str(event.get("previous_event_sha256", "")) != previous:
                raise RuntimeError("campaign event chain is broken or forked")
            seen.add(event_id)
            previous = event_id
            event_type = str(event.get("event_type", ""))
            if event_type in {
                "experiment_campaign_locked",
                "campaign_deviation_appended",
            }:
                kind = (
                    "experiment_campaigns"
                    if event_type == "experiment_campaign_locked"
                    else "campaign_deviations"
                )
                artifact_sha = str(event.get("artifact_sha256", ""))
                artifact_path = self.root / "artifacts" / kind / f"{artifact_sha}.json"
                try:
                    artifact_payload = json.loads(
                        artifact_path.read_text(encoding="utf-8")
                    )
                except (OSError, json.JSONDecodeError) as exc:
                    raise RuntimeError(
                        f"missing or invalid governed artifact: {artifact_path}"
                    ) from exc
                if _sha(artifact_payload) != artifact_sha:
                    raise RuntimeError("governed artifact content digest mismatch")
                if event_type == "experiment_campaign_locked":
                    CampaignLockV1.model_validate(artifact_payload)
                else:
                    CampaignDeviationV1.model_validate(artifact_payload)
        return events

    def write_artifact(self, kind: str, value: BaseModel | dict[str, Any]) -> Path:
        payload = _payload(value)
        digest = _sha(payload)
        path = self.root / "artifacts" / kind / f"{digest}.json"
        data = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        if path.exists():
            if path.read_text(encoding="utf-8") != data:
                raise RuntimeError(f"content-address collision: {path}")
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_new(path, data)
        checksum = {
            "path": str(path.relative_to(self.root)),
            "sha256": hashlib.sha256(data.encode("utf-8")).hexdigest(),
            "content_sha256": digest,
            "written_at": utc_now(),
        }
        self._append_line(
            self.root / "checksums.jsonl", canonical_json(checksum) + "\n"
        )
        return path

    def append_event(
        self,
        event_type: str,
        *,
        experiment_id: str = "",
        status: str = "",
        artifact_sha256: str = "",
        detail: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        events = self.root / "events.jsonl"
        lock_path = self.root / ".events.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            previous = self._last_event_sha(events)
            event = {
                "timestamp": utc_now(),
                "event_type": event_type,
                "campaign_id": self.campaign_id,
                "experiment_id": experiment_id,
                "status": status,
                "artifact_sha256": artifact_sha256,
                "detail": detail or {},
                "previous_event_sha256": previous,
            }
            event["event_id"] = _sha(event)
            self._append_line(events, canonical_json(event) + "\n")
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
        self._append_tsv(event)
        return event

    def status(self) -> dict[str, Any]:
        events_path = self.root / "events.jsonl"
        events = []
        if events_path.exists():
            events = [
                json.loads(line)
                for line in events_path.read_text().splitlines()
                if line
            ]
        return {
            "campaign_id": self.campaign_id,
            "root": str(self.root),
            "event_count": len(events),
            "last_event": events[-1] if events else None,
            "artifacts": sum(1 for _ in (self.root / "artifacts").rglob("*.json"))
            if (self.root / "artifacts").exists()
            else 0,
        }

    @staticmethod
    def _atomic_new(path: Path, data: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        tmp = Path(raw)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(tmp, path)
            except FileExistsError:
                raise
        finally:
            tmp.unlink(missing_ok=True)

    @staticmethod
    def _append_line(path: Path, line: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
        try:
            os.write(fd, line.encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)

    @staticmethod
    def _last_event_sha(path: Path) -> str:
        if not path.exists():
            return ""
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
        return str(json.loads(lines[-1])["event_id"]) if lines else ""

    def _append_tsv(self, event: dict[str, Any]) -> None:
        path = self.root / "results.tsv"
        path.parent.mkdir(parents=True, exist_ok=True)
        new = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=self.EVENT_COLUMNS, delimiter="\t"
            )
            if new:
                writer.writeheader()
            row = {key: event.get(key, "") for key in self.EVENT_COLUMNS}
            row["detail"] = canonical_json(event.get("detail") or {})
            writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())


def loop_result_rows(
    root: Path | str,
    loop_id: str,
    *,
    last: int | None = None,
) -> list[dict[str, Any]]:
    """Derive one compact row per finished experiment from verified event chains."""
    rows: list[dict[str, Any]] = []
    for run in _loop_runs(root, loop_id, last=last):
        campaign = run["campaign"]
        outcome = run["outcome"]
        diagnosis = run["diagnosis"]
        feedback = run["feedback"]
        optimum = feedback.optimum_feedback if feedback is not None else None
        handoff_path = Path(root) / campaign.campaign_id / "cycle_handoff.json"
        handoff: dict[str, Any] = {}
        if handoff_path.is_file():
            try:
                handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                handoff = {}
        rows.append(
            {
                "cycle": campaign.cycle_index,
                "upstream": (campaign.upstream_commit or "")[:8] or "—",
                "integrated": (campaign.integration_commit or "")[:8] or "—",
                "experiment": outcome.experiment_id,
                "params": _metric_text(outcome.metrics, "trainable_params"),
                "primary": _metric_text(outcome.metrics, campaign.primary_metric),
                "metrics": _metrics_text(
                    outcome.metrics,
                    outcome.data_metrics,
                    omit={
                        "trainable_params",
                        campaign.primary_metric,
                        "ship_gates_pass",
                        "gates.pass",
                    },
                ),
                "lean": _lean_text(optimum),
                "gates": _gate_text(outcome.metrics),
                "measurement": _measurement_text(outcome),
                "diagnosis": diagnosis.target if diagnosis is not None else "—",
                "evidence_class": handoff.get("evidence_class", "fixture"),
                "climb": handoff.get("climb_state", "—"),
                "ship": handoff.get("ship_state", "blocked"),
                "status": outcome.status,
                "campaign_id": campaign.campaign_id,
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            int(row["cycle"] or 0),
            str(row["campaign_id"]),
            str(row["experiment"]),
        ),
    )


def _loop_runs(
    root: Path | str, loop_id: str, *, last: int | None
) -> list[dict[str, Any]]:
    root = Path(root)
    campaigns = loop_campaigns(root, loop_id, last=last)
    runs: list[dict[str, Any]] = []
    for campaign in campaigns:
        store = CampaignStore(campaign.campaign_id, root)
        events = store.verify_event_chain()
        for finish in (
            row for row in events if row.get("event_type") == "experiment_finished"
        ):
            experiment_id = str(finish.get("experiment_id", ""))
            outcome = ExperimentOutcome.model_validate(
                _event_payload(store, finish, "outcomes")
            )
            diagnosis_event = next(
                (
                    row
                    for row in reversed(events)
                    if row.get("event_type") == "outcome_diagnosed"
                    and row.get("experiment_id") == experiment_id
                ),
                None,
            )
            feedback_event = next(
                (
                    row
                    for row in reversed(events)
                    if row.get("event_type") == "hypothesizer_feedback_recorded"
                    and row.get("experiment_id") == experiment_id
                ),
                None,
            )
            runs.append(
                {
                    "campaign": campaign,
                    "store": store,
                    "events": events,
                    "outcome": outcome,
                    "diagnosis": (
                        Diagnosis.model_validate(
                            _event_payload(store, diagnosis_event, "diagnoses")
                        )
                        if diagnosis_event is not None
                        else None
                    ),
                    "feedback": (
                        HypothesisFeedback.model_validate(
                            _event_payload(
                                store, feedback_event, "hypothesizer_feedback"
                            )
                        )
                        if feedback_event is not None
                        else None
                    ),
                }
            )
    return sorted(
        runs,
        key=lambda run: (
            int(run["campaign"].cycle_index or 0),
            run["campaign"].campaign_id,
            run["outcome"].experiment_id,
        ),
    )


def loop_campaigns(root: Path, loop_id: str, *, last: int | None) -> list[CampaignSpec]:
    campaigns = sorted(
        (
            CampaignSpec.model_validate_json(path.read_text(encoding="utf-8"))
            for path in root.glob("*/campaign.json")
        ),
        key=lambda item: (item.cycle_index or 0, item.campaign_id),
    )
    campaigns = [item for item in campaigns if item.loop_id == loop_id]
    positions = {
        campaign.campaign_id: index for index, campaign in enumerate(campaigns)
    }
    for expected_cycle, campaign in enumerate(campaigns, start=1):
        if campaign.cycle_index != expected_cycle:
            raise RuntimeError("loop campaign cycles must be unique and contiguous")
        expected_predecessor = (
            campaigns[expected_cycle - 2].campaign_id if expected_cycle > 1 else None
        )
        if campaign.predecessor_campaign_id == expected_predecessor:
            continue
        declared = positions.get(str(campaign.predecessor_campaign_id))
        current = expected_cycle - 1
        if declared is None or declared >= current - 1:
            raise RuntimeError("loop campaign predecessor chain is broken")
        bridge = campaigns[declared + 1 : current]
        prior = campaigns[declared].campaign_id
        if (root / campaign.campaign_id / "cycle_handoff.json").is_file():
            raise RuntimeError("loop campaign predecessor chain is broken")
        for interrupted in bridge:
            if (
                interrupted.predecessor_campaign_id != prior
                or (root / interrupted.campaign_id / "cycle_handoff.json").is_file()
            ):
                raise RuntimeError("loop campaign predecessor chain is broken")
            prior = interrupted.campaign_id
    if last is not None:
        if last < 1:
            raise ValueError("last must be at least one")
        campaigns = campaigns[-last:]
    return campaigns


def loop_diagnostic_rows(
    root: Path | str, loop_id: str, *, last: int | None = None
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in _loop_runs(root, loop_id, last=last):
        campaign = run["campaign"]
        outcome = run["outcome"]
        diagnosis = run["diagnosis"]
        feedback = run["feedback"]
        if diagnosis is not None:
            rows.append(
                {
                    "cycle": campaign.cycle_index,
                    "source": "diagnosis",
                    "area": (
                        f"harness/{diagnosis.harness_family}"
                        if diagnosis.target == "harness"
                        else diagnosis.target
                    ),
                    "signal": "; ".join(diagnosis.evidence) or "—",
                    "authority": "observed",
                    "confidence": f"{diagnosis.confidence:.2f}",
                    "disposition": (
                        diagnosis.recommended_actions[0]
                        if diagnosis.recommended_actions
                        else "monitor"
                    ),
                    "evidence": "—",
                }
            )
        for signal in outcome.harness_signals:
            reproduced = signal.reproduced_on_frozen_input
            rows.append(
                {
                    "cycle": campaign.cycle_index,
                    "source": "harness",
                    "area": signal.family,
                    "signal": signal.code,
                    "authority": "reproduced" if reproduced else "suspected",
                    "confidence": "1.00" if reproduced else "—",
                    "disposition": "repair before run" if reproduced else "observe",
                    "evidence": signal.evidence_uri,
                }
            )
        optimum = feedback.optimum_feedback if feedback is not None else None
        if optimum is None:
            continue
        evidence = (
            f"e:{optimum.metric_evidence_sha256[:8]} "
            f"c:{optimum.metric_certificate_sha256[:8]}"
        )
        if not optimum.breaches:
            rows.append(
                {
                    "cycle": campaign.cycle_index,
                    "source": "Lean",
                    "area": "metric bands",
                    "signal": optimum.policy,
                    "authority": "verified",
                    "confidence": "1.00",
                    "disposition": optimum.policy,
                    "evidence": evidence,
                }
            )
        for breach in optimum.breaches:
            rows.append(
                {
                    "cycle": campaign.cycle_index,
                    "source": "Lean",
                    "area": breach.metric_id,
                    "signal": breach.relation,
                    "authority": breach.authority,
                    "confidence": "1.00",
                    "disposition": optimum.policy,
                    "evidence": evidence,
                }
            )
    unique: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        key = tuple(
            row.get(name)
            for name in ("cycle", "source", "area", "signal", "authority", "evidence")
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def loop_priority_rows(
    root: Path | str, loop_id: str, *, last: int | None = None
) -> list[dict[str, Any]]:
    runs = _loop_runs(root, loop_id, last=last)
    campaigns = {
        campaign.campaign_id: campaign
        for campaign in loop_campaigns(Path(root), loop_id, last=last)
    }
    matrices: list[tuple[CampaignSpec, HypothesisMatrix]] = []
    for campaign in campaigns.values():
        store = CampaignStore(campaign.campaign_id, Path(root))
        for event in store.verify_event_chain():
            if event.get("event_type") != "hypothesis_matrix_formed":
                continue
            matrix = HypothesisMatrix.model_validate(
                _event_payload(store, event, "hypothesis_matrices")
            )
            matrices.append((campaign, matrix))
    predecessor_ids = {
        matrix.predecessor_matrix_id
        for _, matrix in matrices
        if matrix.predecessor_matrix_id is not None
    }
    rows: list[dict[str, Any]] = []
    for campaign, matrix in matrices:
        for priority in sorted(matrix.next_run_priorities, key=lambda item: item.rank):
            rows.append(
                {
                    "cycle": campaign.cycle_index,
                    "rank": priority.rank,
                    "area": priority.area,
                    "hypothesis": priority.hypothesis,
                    "authority": priority.authority,
                    "confidence": f"{priority.confidence:.2f}",
                    "evidence": ",".join(priority.evidence_ids),
                    "experiment": priority.proposed_experiment_id or "—",
                    "action": priority.disposition,
                }
            )
    for run in runs:
        feedback = run["feedback"]
        if feedback is None or feedback.matrix_id in predecessor_ids:
            continue
        campaign = run["campaign"]
        optimum = feedback.optimum_feedback
        if optimum is not None and optimum.policy == "stop":
            rows.append(
                {
                    "cycle": campaign.cycle_index,
                    "rank": 1,
                    "area": "lean_model",
                    "hypothesis": "Repair measurement or the formal model before training.",
                    "authority": "lean_theorem",
                    "confidence": "1.00",
                    "evidence": feedback.feedback_id,
                    "experiment": "—",
                    "action": "stop_campaign",
                }
            )
        elif optimum is not None and optimum.diagnosis_lanes:
            for rank, lane in enumerate(optimum.diagnosis_lanes, start=1):
                rows.append(
                    {
                        "cycle": campaign.cycle_index,
                        "rank": rank,
                        "area": lane,
                        "hypothesis": f"Test the {lane} lane without assigning causality.",
                        "authority": "lean_assumption",
                        "confidence": "—",
                        "evidence": feedback.feedback_id,
                        "experiment": "—",
                        "action": "block_promotion",
                    }
                )
        elif feedback.diagnosis_target == "harness":
            rows.append(
                {
                    "cycle": campaign.cycle_index,
                    "rank": 1,
                    "area": feedback.harness_family,
                    "hypothesis": "Repair the reproduced canonical harness failure.",
                    "authority": "reproduced_harness_signal",
                    "confidence": "1.00",
                    "evidence": feedback.feedback_id,
                    "experiment": "—",
                    "action": "repair_before_run",
                }
            )
    if not rows:
        return []
    latest_cycle = max(int(row.get("cycle") or 0) for row in rows)
    return [row for row in rows if int(row.get("cycle") or 0) == latest_cycle]


def render_loop_result_matrix(
    root: Path | str, loop_id: str, *, last: int | None = None
) -> str:
    """Render canonical liveness, result, diagnostic, and priority views."""
    result_columns = (
        ("cycle", "Cycle"),
        ("upstream", "Upstream"),
        ("integrated", "Integrated"),
        ("experiment", "Experiment"),
        ("params", "Params"),
        ("primary", "Primary"),
        ("metrics", "Metrics"),
        ("lean", "Lean bands"),
        ("gates", "Gates"),
        ("measurement", "Measurement"),
        ("diagnosis", "Diagnosis"),
        ("evidence_class", "Evidence"),
        ("climb", "Climb"),
        ("ship", "Ship"),
        ("status", "Status"),
    )
    diagnostic_columns = (
        ("cycle", "Cycle"),
        ("source", "Source"),
        ("area", "Area"),
        ("signal", "Signal"),
        ("authority", "Authority"),
        ("confidence", "Confidence"),
        ("disposition", "Disposition"),
        ("evidence", "Evidence"),
    )
    priority_columns = (
        ("cycle", "Cycle"),
        ("rank", "Rank"),
        ("area", "Area"),
        ("hypothesis", "Hypothesis"),
        ("authority", "Authority"),
        ("confidence", "Confidence"),
        ("evidence", "Evidence"),
        ("experiment", "Experiment"),
        ("action", "Action"),
    )
    state_path = Path(root) / "loops" / loop_id / "state.json"
    state: dict[str, Any] = {}
    if state_path.is_file():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            state = {}
    recorded_state = str(state.get("state") or "UNKNOWN")
    driver_pid = state.get("pid")
    if recorded_state == "RUNNING" and not _pid_exists(driver_pid):
        recorded_state = "DEAD"
    heartbeat = str(state.get("heartbeat_at") or "—")
    if heartbeat != "—" and recorded_state not in {"BLOCKED", "DEAD"}:
        try:
            stamp = datetime.fromisoformat(heartbeat.replace("Z", "+00:00"))
            age = (datetime.now(timezone.utc) - stamp).total_seconds()
            if age > 2 * MAX_RUN_SECONDS:
                recorded_state = "STALE"
        except ValueError:
            recorded_state = "STALE"
    liveness_columns = (
        ("state", "State"),
        ("phase", "Phase"),
        ("cycle", "Cycle"),
        ("active", "Active"),
        ("last", "Last complete"),
        ("next", "Next action"),
        ("blockers", "Blockers"),
        ("pid", "Driver PID"),
        ("stage", "Active stage"),
        ("child", "Child PID"),
        ("heartbeat", "Heartbeat"),
    )
    liveness_rows = [
        {
            "state": recorded_state,
            "phase": state.get("phase", "—"),
            "cycle": state.get("cycle_index", "—"),
            "active": state.get("active_campaign_id") or "—",
            "last": state.get("last_completed_campaign_id") or "—",
            "next": state.get("next_action") or "—",
            "blockers": state.get("blocker_count", 0),
            "pid": driver_pid or "—",
            "stage": state.get("active_stage") or "—",
            "child": state.get("child_pid") or "—",
            "heartbeat": heartbeat,
        }
    ]
    return "\n\n".join(
        (
            _render_table("Liveness", liveness_columns, liveness_rows),
            _render_table(
                "Run Results",
                result_columns,
                loop_result_rows(root, loop_id, last=last),
            ),
            _render_table(
                "Diagnostic Signals",
                diagnostic_columns,
                loop_diagnostic_rows(root, loop_id, last=last),
            ),
            _render_table(
                "Speculative Next-Run Priorities",
                priority_columns,
                loop_priority_rows(root, loop_id, last=last),
            ),
        )
    )


def _render_table(
    title: str,
    columns: tuple[tuple[str, str], ...],
    rows: list[dict[str, Any]],
) -> str:
    lines = [
        title,
        "| " + " | ".join(label for _, label in columns) + " |",
        "|" + "|".join(" --- " for _ in columns) + "|",
    ]
    if not rows:
        rows = [{key: "—" for key, _ in columns}]
    lines.extend(
        "| "
        + " | ".join(_markdown_cell(row.get(key, "—")) for key, _ in columns)
        + " |"
        for row in rows
    )
    return "\n".join(lines)


def _pid_exists(pid: object) -> bool:
    if not isinstance(pid, int) or pid < 1:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _measurement_text(outcome: ExperimentOutcome) -> str:
    if outcome.status != "completed":
        return "incomplete"
    if any(
        stage.get("measurement_complete") is False for stage in outcome.stage_telemetry
    ):
        return "incomplete"
    if evaluation_measurement_incomplete(outcome.metrics):
        return "incomplete"
    return "complete"


def _event_artifact(
    store: CampaignStore,
    event: dict[str, Any],
    kind: str,
) -> Path:
    digest = str(event.get("artifact_sha256", ""))
    return store.root / "artifacts" / kind / f"{digest}.json"


def _event_payload(
    store: CampaignStore,
    event: dict[str, Any],
    kind: str,
) -> dict[str, Any]:
    digest = str(event.get("artifact_sha256", ""))
    payload = json.loads(
        _event_artifact(store, event, kind).read_text(encoding="utf-8")
    )
    if _sha(payload) != digest:
        raise RuntimeError(f"{kind} artifact content digest mismatch")
    return payload


def _metric_text(metrics: dict[str, float], name: str) -> str:
    for key, value in metrics.items():
        if key == name or key.endswith(f".{name}"):
            return f"{value:g}"
    return "—"


def _metrics_text(
    metrics: dict[str, float],
    data_metrics: dict[str, float],
    *,
    omit: set[str],
) -> str:
    def omitted(name: str) -> bool:
        return any(name == item or name.endswith(f".{item}") for item in omit)

    allow = {
        "n",
        "meaningful_program_rate",
        "structural_similarity",
        "binder_reference_f1",
        "parse_rate",
        "latency_ms_p50",
        "ast_beq_rate",
        "canonical_beq_rate",
    }

    def allowed(name: str) -> bool:
        return name.split(".")[-1] in allow

    values = [
        f"{name}={value:g}"
        for name, value in sorted(metrics.items())
        if not omitted(name) and allowed(name)
    ]
    values.extend(
        f"data.{name}={value:g}"
        for name, value in sorted(data_metrics.items())
        if allowed(name)
    )
    return ", ".join(values) or "—"


def _lean_text(optimum: Any | None) -> str:
    if optimum is None:
        return "—"
    if not optimum.breaches:
        return optimum.policy
    return ", ".join(
        f"{item.metric_id}:{item.relation}[{item.authority}]"
        for item in optimum.breaches
    )


def _gate_text(metrics: dict[str, float]) -> str:
    for name in ("ship_gates_pass", "gates.pass"):
        value = _metric_text(metrics, name)
        if value != "—":
            return "pass" if float(value) else "fail"
    return "—"


def _markdown_cell(value: Any) -> str:
    return str(value).replace("|", r"\|").replace("\n", " ")
