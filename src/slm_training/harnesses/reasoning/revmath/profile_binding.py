"""HARN-10 / SLM-548: bind ``reasoning/revmath`` runs to canonical campaign owners.

Locks ``ExperimentCampaignV1`` through ``CampaignStore``, rejects unlocked or
inconsistent task bindings, optionally validates formal-preflight obligations,
executes frozen tasks via ``run_revmath_task``, and persists evidence through
existing decode-stats / replay / disposition / formal artifact paths — never a
profile-local campaign DB or evidence store.

Module name deliberately avoids ``revmath*campaign`` (owner-map shadow ban).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slm_training.autoresearch.campaign_formal_evidence import (
    decision_support_report,
    empirical_status_from_result,
    formal_status_from_authority,
    ledger_axis_statuses,
    link_four_axis_and_revmath,
    revmath_report_digest,
)
from slm_training.autoresearch.experiment_campaign import (
    ArtifactRequirementV1,
    CampaignArmV1,
    CampaignControlV1,
    CampaignEndpointV1,
    CampaignGateV1,
    CampaignLockV1,
    CampaignResultV1,
    ExperimentCampaignV1,
    MultiplicityFamilyV1,
    campaign_manifest_sha256,
)
from slm_training.autoresearch.formal import (
    formal_preflight_sha256,
    migrate_formal_preflight_v1,
    validate_formal_preflights,
)
from slm_training.autoresearch.schemas import CampaignBudget, CampaignSpec, FormalPreflightV1
from slm_training.autoresearch.storage import CampaignStore
from slm_training.harness_core.lineage.records import content_sha
from slm_training.harnesses.experiments.mechanism_disposition_report import (
    build_mechanism_disposition_record,
)
from slm_training.harnesses.experiments.slm160_spv_disposition import Disposition
from slm_training.harnesses.reasoning.profiles import REVMATH_PROFILE_ID
from slm_training.harnesses.reasoning.revmath.report import build_revmath_report
from slm_training.harnesses.reasoning.revmath.runner import RevmathRunRecord, run_revmath_task
from slm_training.harnesses.reasoning.revmath.schemas import (
    RevmathCampaignBindingV1,
    RevmathResultV1,
    RevmathSchemaError,
    RevmathTaskV1,
    parse_revmath_task,
)
from slm_training.levers import MAX_RUN_MINUTES
from slm_training.models.decode_stats import (
    DecodeIdentityV1,
    DecodeStats,
    DecodeStatsRecordV1,
    append_decode_stats_record,
    build_decode_stats_record,
)
from slm_training.runtime.telemetry.replay_bundle import (
    ReplayBundleV1,
    append_replay_bundle,
)

__all__ = [
    "RevmathCampaignBindingError",
    "RevmathProfilePlanV1",
    "RevmathProfileRunV1",
    "assert_tasks_match_lock",
    "bind_tasks_to_lock",
    "load_experiment_campaign",
    "materialize_fixture_campaign",
    "persist_formal_preflight",
    "replay_profile_run_from_store",
    "require_locked_campaign",
    "run_revmath_profile",
]

_EVENT_PROFILE_COMPLETED = "revmath_profile_run_completed"
_EVENT_TASK_COMPLETED = "revmath_task_run_completed"
_MECHANISM_ID = "reasoning.revmath.profile"


class RevmathCampaignBindingError(ValueError):
    """Unlocked, mismatched, or formally inconsistent revmath campaign binding."""


@dataclass(frozen=True)
class RevmathProfilePlanV1:
    """Inputs for materializing a fixture/wiring ``ExperimentCampaignV1``."""

    campaign_id: str
    experiment_id: str
    source_commit: str = "0" * 40
    source_dirty: bool = False
    author: str = "revmath-profile"
    claim_class: str = "fixture"
    created_at: str = "1970-01-01T00:00:00Z"
    max_wall_minutes: float = float(MAX_RUN_MINUTES)
    hypothesis: str = (
        "Locked ExperimentCampaignV1 binding makes revmath runs replayable "
        "in the canonical campaign lineage."
    )
    decision: str = (
        "Whether reasoning/revmath rejects unlocked or inconsistent campaign "
        "bindings before execution."
    )

    def __post_init__(self) -> None:
        if self.claim_class not in {"fixture", "wiring", "diagnostic"}:
            raise RevmathCampaignBindingError(
                "profile materialize only exposes fixture/wiring/diagnostic claim_class"
            )
        if self.max_wall_minutes <= 0 or self.max_wall_minutes > MAX_RUN_MINUTES:
            raise RevmathCampaignBindingError("max_wall_minutes must obey MAX_RUN_MINUTES")
        if len(self.source_commit) != 40 or any(
            ch not in "0123456789abcdef" for ch in self.source_commit
        ):
            raise RevmathCampaignBindingError("source_commit must be 40 lowercase hex chars")


@dataclass(frozen=True)
class RevmathProfileRunV1:
    """One locked profile execution recorded through ``CampaignStore``."""

    profile_id: str
    campaign_id: str
    experiment_id: str
    manifest_sha256: str
    claim_class: str
    tasks: tuple[RevmathTaskV1, ...]
    runs: tuple[RevmathRunRecord, ...]
    report: dict[str, Any]
    disposition: dict[str, Any]
    formal_preflight_sha256s: tuple[str, ...]
    campaign_result: dict[str, Any]
    store_root: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "revmath_profile_run/v1",
            "profile_id": self.profile_id,
            "campaign_id": self.campaign_id,
            "experiment_id": self.experiment_id,
            "manifest_sha256": self.manifest_sha256,
            "claim_class": self.claim_class,
            "tasks": [task.model_dump(mode="json") for task in self.tasks],
            "runs": [run.to_dict() for run in self.runs],
            "report": self.report,
            "disposition": self.disposition,
            "formal_preflight_sha256s": list(self.formal_preflight_sha256s),
            "campaign_result": self.campaign_result,
            "store_root": self.store_root,
        }


class _JsonlReplayStore:
    """Minimal append-only store satisfying ``ReplayBundleStore``."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, row: dict[str, Any]) -> str:
        digest = content_sha(row)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        return digest

    def iter_kind(self, kind: str):
        if not self.path.is_file():
            return
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("kind") == kind:
                    yield row


def load_experiment_campaign(path: Path | str) -> ExperimentCampaignV1:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return ExperimentCampaignV1.model_validate(raw)


def materialize_fixture_campaign(
    plan: RevmathProfilePlanV1,
    tasks: Sequence[RevmathTaskV1],
) -> ExperimentCampaignV1:
    """Build a fixture/wiring campaign from frozen task budgets and arm ids."""

    if not tasks:
        raise RevmathCampaignBindingError("at least one task is required to materialize")
    campaign_ids = {task.campaign.campaign_id for task in tasks}
    if len(campaign_ids) != 1 or plan.campaign_id not in campaign_ids:
        raise RevmathCampaignBindingError(
            "all tasks must share plan.campaign_id before materialize"
        )
    experiment_ids = {task.campaign.experiment_id for task in tasks}
    if plan.experiment_id not in experiment_ids and len(experiment_ids) != 1:
        # Allow rebinding experiment_id at materialize time when tasks still
        # carry placeholder experiment ids from hermetic fixtures.
        pass

    locked_arms = sorted(
        {arm for task in tasks for arm in task.campaign.locked_arm_ids}
    )
    if not locked_arms:
        raise RevmathCampaignBindingError("tasks declare no locked_arm_ids")

    config_sha = content_sha(
        {
            "profile_id": REVMATH_PROFILE_ID,
            "campaign_id": plan.campaign_id,
            "experiment_id": plan.experiment_id,
            "task_identity_digests": [task.identity_digest() for task in tasks],
            "task_canonical_hashes": [task.canonical_hash() for task in tasks],
        }
    )
    control_arm = locked_arms[0]
    candidate_arm = "revmath.candidate"
    if candidate_arm in locked_arms and control_arm == candidate_arm:
        raise RevmathCampaignBindingError("control and candidate arms collide")

    max_wall = max(task.budget.max_wall_seconds for task in tasks) / 60.0
    wall_minutes = min(plan.max_wall_minutes, max(max_wall, 0.05))
    stopping = sorted(
        {
            rule
            for task in tasks
            for rule in task.budget.stopping_rule_ids
        }
        | {"revmath_profile_budget_wall"}
    )

    return ExperimentCampaignV1(
        campaign_id=plan.campaign_id,
        experiment_id=plan.experiment_id,
        hypothesis=plan.hypothesis,
        decision=plan.decision,
        endpoints=(
            CampaignEndpointV1(
                endpoint_id="tasks_complete",
                metric="tasks_run",
                role="primary",
                direction="increase",
                minimum_effect=0.0,
            ),
        ),
        arms=(
            CampaignArmV1(arm_id=control_arm, role="control", config_sha256=config_sha),
            CampaignArmV1(
                arm_id=candidate_arm, role="candidate", config_sha256=config_sha
            ),
        ),
        seeds=(0,),
        budget=CampaignBudget(
            max_experiments=max(2, len(tasks)),
            max_wall_minutes=wall_minutes,
        ),
        stopping_rules=tuple(stopping),
        controls=(
            CampaignControlV1(
                control_id=control_arm,
                description=(
                    "Hermetic/control arm for reasoning/revmath profile binding; "
                    "never a measured promotion default."
                ),
                kind="negative",
            ),
        ),
        negative_controls=(control_arm,),
        multiplicity_families=(
            MultiplicityFamilyV1(
                family_id="revmath_fixture",
                hypothesis_ids=("tasks_complete",),
                alpha=0.05,
            ),
        ),
        promotion_gates=(
            CampaignGateV1(
                gate_id="tasks_ok",
                endpoint_id="tasks_complete",
                operator="ge",
                threshold=float(len(tasks)),
            ),
        ),
        rollback_gates=(
            CampaignGateV1(
                gate_id="tasks_fail",
                endpoint_id="tasks_complete",
                operator="lt",
                threshold=float(len(tasks)),
            ),
        ),
        artifact_requirements=(ArtifactRequirementV1(kind="version_stamp"),),
        formal_obligations=(),
        claim_class=plan.claim_class,  # type: ignore[arg-type]
        source_commit=plan.source_commit,
        source_dirty=plan.source_dirty,
        author=plan.author,
        created_at=plan.created_at,
    )


def bind_tasks_to_lock(
    tasks: Sequence[RevmathTaskV1],
    lock: CampaignLockV1,
) -> tuple[RevmathTaskV1, ...]:
    """Rewrite campaign binding fields to the locked manifest (materialize path)."""

    manifest = lock.manifest
    arm_ids = {arm.arm_id for arm in manifest.arms}
    bound: list[RevmathTaskV1] = []
    for task in tasks:
        unknown = set(task.campaign.locked_arm_ids) - arm_ids
        if unknown:
            raise RevmathCampaignBindingError(
                f"task {task.task_id} locks unknown arms: {sorted(unknown)}"
            )
        if task.campaign.campaign_id != manifest.campaign_id:
            raise RevmathCampaignBindingError(
                f"task {task.task_id} campaign_id {task.campaign.campaign_id!r} "
                f"!= locked {manifest.campaign_id!r}"
            )
        rebound = task.model_copy(
            update={
                "campaign": RevmathCampaignBindingV1(
                    campaign_id=manifest.campaign_id,
                    experiment_id=manifest.experiment_id,
                    campaign_manifest_sha256=lock.manifest_sha256,
                    locked_arm_ids=task.campaign.locked_arm_ids,
                )
            }
        )
        bound.append(rebound)
    return tuple(bound)


def assert_tasks_match_lock(
    tasks: Sequence[RevmathTaskV1],
    lock: CampaignLockV1,
) -> None:
    """Fail closed when any task binding disagrees with the locked manifest."""

    manifest = lock.manifest
    arm_ids = {arm.arm_id for arm in manifest.arms}
    for task in tasks:
        binding = task.campaign
        if binding.campaign_id != manifest.campaign_id:
            raise RevmathCampaignBindingError(
                f"task {task.task_id} campaign_id mismatch "
                f"({binding.campaign_id} != {manifest.campaign_id})"
            )
        if binding.experiment_id != manifest.experiment_id:
            raise RevmathCampaignBindingError(
                f"task {task.task_id} experiment_id mismatch "
                f"({binding.experiment_id} != {manifest.experiment_id})"
            )
        if binding.campaign_manifest_sha256 != lock.manifest_sha256:
            raise RevmathCampaignBindingError(
                f"task {task.task_id} campaign_manifest_sha256 mismatch with lock "
                f"({binding.campaign_manifest_sha256} != {lock.manifest_sha256})"
            )
        unknown = set(binding.locked_arm_ids) - arm_ids
        if unknown:
            raise RevmathCampaignBindingError(
                f"task {task.task_id} locks unknown arms: {sorted(unknown)}"
            )


def require_locked_campaign(
    store: CampaignStore,
    experiment_id: str,
) -> CampaignLockV1:
    """Load the authoritative lock or reject unlocked execution."""

    try:
        return store.load_experiment_campaign(experiment_id)
    except FileNotFoundError as exc:
        raise RevmathCampaignBindingError(
            f"experiment {experiment_id!r} is unlocked; refuse revmath execution"
        ) from exc


def persist_formal_preflight(
    store: CampaignStore,
    preflight: FormalPreflightV1 | Mapping[str, Any],
) -> str:
    """Write FormalPreflight + formal_authority/v2 through CampaignStore.

    v1 stays readable via migrate adapters; the preferred write envelope is
    EVID-06 ``formal_authority/v2`` (no parallel evidence store).
    """

    from slm_training.formal.authority import (
        from_formal_preflight_v1,
        prefer_write_payload,
    )

    if isinstance(preflight, Mapping):
        preflight = migrate_formal_preflight_v1(preflight)
    digest = formal_preflight_sha256(preflight)
    path = store.write_artifact("formal_preflights", preflight)
    authority = from_formal_preflight_v1(preflight, v1_content_digest=digest)
    auth_path = store.write_artifact(
        "formal_authority", prefer_write_payload(authority)
    )
    store.append_event(
        "formal_preflight_bound",
        experiment_id=preflight.experiment_id,
        status=preflight.status,
        artifact_sha256=path.stem,
        detail={
            "formal_preflight_sha256": digest,
            "formal_authority_sha256": auth_path.stem,
            "formal_authority_schema": authority.schema,
        },
    )
    return digest


def _validate_formal_if_required(
    store: CampaignStore,
    lock: CampaignLockV1,
    *,
    experiment_spec: Any | None,
) -> tuple[str, ...]:
    """Reject when lock formal obligations are present but inconsistent.

    Fixture/wiring campaigns may declare zero obligations. When obligations
    exist, require matching ``formal_preflights/`` artifacts via the canonical
    ``validate_formal_preflights`` owner.
    """

    obligations = lock.manifest.formal_obligations
    if not obligations:
        return ()
    if experiment_spec is None:
        raise RevmathCampaignBindingError(
            "cannot execute: formal_obligations locked without experiment_spec"
        )
    preflights = validate_formal_preflights(store.root, experiment_spec, lock.manifest)
    return tuple(formal_preflight_sha256(item) for item in preflights)


def _persist_run_evidence(
    store: CampaignStore,
    lock: CampaignLockV1,
    record: RevmathRunRecord,
    *,
    prev_decode_digest: str | None,
) -> str | None:
    """Persist result/replay/decode-stats via canonical owners; return new digest."""

    run_path = store.write_artifact("revmath_runs", record.to_dict())
    store.append_event(
        _EVENT_TASK_COMPLETED,
        experiment_id=lock.manifest.experiment_id,
        status=str(record.result.judgment.get("outcome", "unknown")),
        artifact_sha256=run_path.stem,
        detail={
            "task_id": record.task.task_id,
            "task_identity_digest": record.task.identity_digest(),
            "result_id": record.result.result_id,
            "replay_bundle_hash": record.replay_bundle.bundle_hash,
            "campaign_manifest_sha256": lock.manifest_sha256,
        },
    )
    store.write_artifact("replay_bundles", record.replay_bundle.to_dict())
    append_replay_bundle(
        _JsonlReplayStore(store.root / "replay_bundles.jsonl"),
        record.replay_bundle,
    )

    raw = DecodeStatsRecordV1.from_dict(record.decode_stats_record)
    decode = build_decode_stats_record(
        DecodeStats.from_dict(raw.stats),
        measurement_stage=raw.measurement_stage,
        completeness=raw.completeness,
        identity=DecodeIdentityV1.from_dict(raw.identity),
        legal_domain_size=raw.legal_domain_size,
        legal_domain_status=raw.legal_domain_status,
        witness_ids=raw.witness_ids,
        prev_record_digest=prev_decode_digest,
    )
    append_decode_stats_record(decode, store.root / "decode_stats.jsonl")
    store.write_artifact("decode_stats_records", decode.to_dict())
    return decode.record_digest


def _build_disposition(
    *,
    claim_class: str,
    manifest_sha256: str,
    runs: Sequence[RevmathRunRecord],
) -> dict[str, Any]:
    activation = content_sha(
        {
            "manifest_sha256": manifest_sha256,
            "result_ids": [run.result.result_id for run in runs],
            "bundle_hashes": [run.replay_bundle.bundle_hash for run in runs],
        }
    )
    record = build_mechanism_disposition_record(
        mechanism_id=_MECHANISM_ID,
        owner_module="slm_training.harnesses.reasoning.revmath.profile_binding",
        owner_version="harn10/v1",
        default_state="n/a" if claim_class == "fixture" else "off",
        authority_class="evaluation-only",
        evidence_cutoff=manifest_sha256,
        evidence_class="fixture" if claim_class in {"fixture", "wiring"} else "scratch",
        disposition=Disposition.RETAIN_DIAGNOSTIC,
        rationale=(
            "reasoning/revmath profile run recorded under locked ExperimentCampaignV1; "
            "fixture/wiring evidence cannot adopt a production default."
        ),
        activation_evidence=activation,
        applicable_packs=(REVMATH_PROFILE_ID,),
        matched_controls=True,
        semantic_effect="none",
        safety_invariants=(
            "never_bypass_experiment_campaign_v1",
            "never_parallel_revmath_evidence_store",
        ),
        issues=("SLM-548", "HARN-10"),
    )
    return record.to_dict()


def run_revmath_profile(
    tasks: Sequence[RevmathTaskV1] | Sequence[Mapping[str, Any]],
    *,
    store_root: Path,
    manifest: ExperimentCampaignV1 | None = None,
    plan: RevmathProfilePlanV1 | None = None,
    hermetic: bool = True,
    lean_root: Path | None = None,
    experiment_spec: Any | None = None,
    rebind_tasks: bool = False,
) -> RevmathProfileRunV1:
    """Lock (or require) ExperimentCampaignV1, bind tasks, run, persist evidence.

    * ``manifest`` — load/lock this campaign (strict binding unless ``rebind_tasks``).
    * ``plan`` — materialize a fixture campaign then rebind tasks to the lock.
    Exactly one of ``manifest`` / ``plan`` is required.
    """

    parsed: list[RevmathTaskV1] = []
    for item in tasks:
        if isinstance(item, RevmathTaskV1):
            parsed.append(item)
        else:
            parsed.append(parse_revmath_task(item))
    if not parsed:
        raise RevmathCampaignBindingError("no tasks provided")

    if (manifest is None) == (plan is None):
        raise RevmathCampaignBindingError("provide exactly one of manifest or plan")

    if plan is not None:
        manifest = materialize_fixture_campaign(plan, parsed)
        rebind_tasks = True

    assert manifest is not None
    store = CampaignStore(manifest.campaign_id, store_root)
    store.initialize(
        CampaignSpec(
            campaign_id=manifest.campaign_id,
            objective=manifest.decision,
            primary_metric=manifest.endpoints[0].metric,
            track="twotower",
            min_hypotheses=5,
            budget=manifest.budget,
            created_at=manifest.created_at,
            notes=f"profile={REVMATH_PROFILE_ID}",
        )
    )
    lock = store.lock_experiment_campaign(manifest)
    # Fail closed if a caller tries to run without going through lock (reload).
    lock = require_locked_campaign(store, manifest.experiment_id)
    if lock.manifest_sha256 != campaign_manifest_sha256(manifest):
        raise RevmathCampaignBindingError("locked digest disagrees with provided manifest")

    bound_tasks: tuple[RevmathTaskV1, ...]
    if rebind_tasks:
        bound_tasks = bind_tasks_to_lock(parsed, lock)
    else:
        assert_tasks_match_lock(parsed, lock)
        bound_tasks = tuple(parsed)

    formal_digests = _validate_formal_if_required(
        store, lock, experiment_spec=experiment_spec
    )

    runs: list[RevmathRunRecord] = []
    prev_decode: str | None = None
    for task in bound_tasks:
        # Re-check lock before every task (reject mid-run unlock/tamper).
        live = require_locked_campaign(store, lock.manifest.experiment_id)
        if live.manifest_sha256 != lock.manifest_sha256:
            raise RevmathCampaignBindingError("campaign lock changed during profile run")
        assert_tasks_match_lock((task,), live)
        record = run_revmath_task(task, lean_root=lean_root, hermetic=hermetic)
        prev_decode = _persist_run_evidence(
            store, live, record, prev_decode_digest=prev_decode
        )
        runs.append(record)

    report = build_revmath_report(
        report_id=f"report.{lock.manifest.experiment_id}",
        tasks=list(bound_tasks),
        results=[run.result for run in runs],
    )
    disposition = _build_disposition(
        claim_class=lock.manifest.claim_class,
        manifest_sha256=lock.manifest_sha256,
        runs=runs,
    )
    store.write_artifact("mechanism_dispositions", disposition)

    formal_status = "absent"
    authority_digests: list[str] = []
    ledger_digest = None
    axis_statuses = ledger_axis_statuses(None)
    remainder_ids: list[str] = []
    # Prefer latest formal_authority/v2 artifact when present (EVID-06).
    auth_dir = store.root / "artifacts" / "formal_authority"
    if auth_dir.is_dir():
        auth_files = sorted(auth_dir.glob("*.json"))
        if auth_files:
            auth_payload = json.loads(auth_files[-1].read_text(encoding="utf-8"))
            formal_status = formal_status_from_authority(auth_payload)
            authority_digests.append(auth_files[-1].stem)
            # four_axis digest may be on the authority envelope
            ledger_digest = auth_payload.get("four_axis_ledger_sha256")
            remainder_ids = [
                str(x) for x in auth_payload.get("empirical_remainder_claim_ids", ())
            ]
    report_digest = revmath_report_digest(report)
    result_digests = tuple(
        content_sha(run.result.model_dump(mode="json")) for run in runs
    )
    failed_runs = False
    for run in runs:
        judgment = run.result.judgment
        payload = judgment.to_dict() if hasattr(judgment, "to_dict") else judgment
        if isinstance(payload, dict):
            outcome = str(payload.get("outcome", "")).lower()
            # KERN-01: witnessed == successful formal observation; refuted/unknown/invalid fail.
            if outcome not in {"witnessed", "proved", "proven", "certified"}:
                failed_runs = True
                break
    empirical_status = empirical_status_from_result(
        ship_gates_passed=False,
        exploratory=lock.manifest.claim_class in {"fixture", "wiring", "diagnostic"},
        claim_class=lock.manifest.claim_class,
        endpoint_success=False if failed_runs else (True if runs else None),
        runs_failed=failed_runs,
    )
    # Fixture/wiring profile runs stay exploratory even when tasks prove.
    if lock.manifest.claim_class in {"fixture", "wiring", "diagnostic"}:
        empirical_status = "exploratory"
    formal_empirical = link_four_axis_and_revmath(
        formal_status=formal_status,  # type: ignore[arg-type]
        empirical_status=empirical_status,  # type: ignore[arg-type]
        formal_authority_digests=authority_digests,
        formal_preflight_digests=formal_digests,
        four_axis_ledger_sha256=ledger_digest,
        revmath_report_sha256=report_digest,
        revmath_result_digests=result_digests,
        empirical_remainder_claim_ids=remainder_ids,
        assumption_axis_status=axis_statuses["assumption_strength"],
        computability_axis_status=axis_statuses["computability"],
        resource_bounds_axis_status=axis_statuses["resource_bounds"],
        refinement_axis_status=axis_statuses["implementation_refinement"],
        notes=(
            "INTEG-03 linked revmath profile evidence; formal status does not "
            "authorize ship/promotion."
        ),
    )
    # Persist decision-support report beside campaign results (query surface).
    support_report = decision_support_report(formal_empirical)
    store.write_artifact("campaign_decision_support_reports", support_report)

    campaign_result = CampaignResultV1(
        campaign_id=lock.manifest.campaign_id,
        experiment_id=lock.manifest.experiment_id,
        manifest_sha256=lock.manifest_sha256,
        claim_class=lock.manifest.claim_class,
        arm_seed_results=tuple(
            (arm, 0) for arm in sorted({a for t in bound_tasks for a in t.campaign.locked_arm_ids})
        ),
        endpoint_values={"tasks_complete": float(len(runs))},
        artifacts=(),
        exploratory=lock.manifest.claim_class in {"fixture", "wiring", "diagnostic"},
        ship_gates_passed=False,
        formal_empirical=formal_empirical,
    )
    result_path = store.write_artifact("campaign_results", campaign_result)
    profile = RevmathProfileRunV1(
        profile_id=REVMATH_PROFILE_ID,
        campaign_id=lock.manifest.campaign_id,
        experiment_id=lock.manifest.experiment_id,
        manifest_sha256=lock.manifest_sha256,
        claim_class=lock.manifest.claim_class,
        tasks=bound_tasks,
        runs=tuple(runs),
        report=report.model_dump(mode="json"),
        disposition=disposition,
        formal_preflight_sha256s=formal_digests,
        campaign_result=campaign_result.model_dump(mode="json"),
        store_root=str(store.root),
    )
    summary_path = store.write_artifact("revmath_profile_runs", profile.to_dict())
    store.append_event(
        _EVENT_PROFILE_COMPLETED,
        experiment_id=lock.manifest.experiment_id,
        status="completed",
        artifact_sha256=summary_path.stem,
        detail={
            "manifest_sha256": lock.manifest_sha256,
            "result_artifact_sha256": result_path.stem,
            "task_count": len(runs),
            "formal_preflight_sha256s": list(formal_digests),
            "disposition_hash": disposition.get("record_hash"),
        },
    )
    return profile


def replay_profile_run_from_store(
    store_root: Path,
    campaign_id: str,
    *,
    experiment_id: str,
) -> dict[str, Any]:
    """Reload the latest profile-run artifact and verify replay bundle integrity."""

    from slm_training.harnesses.reasoning.revmath.replay import (
        assert_revmath_replay_integrity,
    )

    store = CampaignStore(campaign_id, store_root)
    lock = require_locked_campaign(store, experiment_id)
    events = store.verify_event_chain()
    completed = [
        row
        for row in events
        if row.get("event_type") == _EVENT_PROFILE_COMPLETED
        and row.get("experiment_id") == experiment_id
    ]
    if not completed:
        raise RevmathCampaignBindingError("no revmath profile completion event to replay")
    artifact_sha = str(completed[-1].get("artifact_sha256", ""))
    path = store.root / "artifacts" / "revmath_profile_runs" / f"{artifact_sha}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("manifest_sha256") != lock.manifest_sha256:
        raise RevmathCampaignBindingError("replay artifact manifest digest mismatch")
    for run in payload.get("runs", []):
        bundle = ReplayBundleV1.from_dict(run["replay_bundle"])
        assert_revmath_replay_integrity(bundle)
        task = parse_revmath_task(run["task"])
        result = RevmathResultV1.model_validate(run["result"])
        if result.task_identity_digest != task.identity_digest():
            raise RevmathSchemaError("replay result task_identity_digest mismatch")
        if task.campaign.campaign_manifest_sha256 != lock.manifest_sha256:
            raise RevmathCampaignBindingError(
                "replay task campaign binding disagrees with lock"
            )
    return payload
