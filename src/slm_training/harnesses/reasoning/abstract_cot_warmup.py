"""Resumable, bounded Abstract-CoT warm-up orchestration (SLM-311).

The loop owns campaign locking and append-only iteration evidence.  A caller
supplies the AP-020 collector/AP-019 trainer binding; this deliberately avoids
creating a second training path or treating fixture policies as checkpoints.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from slm_training.autoresearch.experiment_campaign import (
    ArtifactRequirementV1,
    CampaignArmV1,
    CampaignControlV1,
    CampaignEndpointV1,
    CampaignGateV1,
    ExperimentCampaignV1,
    MultiplicityFamilyV1,
)
from slm_training.autoresearch.schemas import CampaignBudget, CampaignSpec
from slm_training.autoresearch.storage import CampaignStore
from slm_training.harness_core.lineage.records import content_sha
from slm_training.levers import MAX_RUN_MINUTES

__all__ = [
    "AbstractWarmupCampaignV1",
    "IterationRecord",
    "WarmupResult",
    "run_abstract_cot_warmup",
]

_EVENT = "abstract_cot_warmup_iteration_completed"
_PHASE_FIELDS = frozenset(
    {"command", "config_sha256", "input_sha256", "output_sha256", "token_count", "result_ref"}
)
_REQUIRED_PHASES = frozenset({"ap020_collection", "ap019_segment_masked_sft"})


@dataclass(frozen=True)
class AbstractWarmupCampaignV1:
    campaign_id: str
    iterations: int = 3
    seed: int = 0
    source_commit: str = "0" * 40
    claim_class: str = "fixture"
    max_wall_minutes: float = float(MAX_RUN_MINUTES)

    def __post_init__(self) -> None:
        if self.iterations < 1:
            raise ValueError("iterations must be >= 1")
        if self.claim_class != "fixture":
            raise ValueError("SLM-311 only exposes the fixture wiring campaign")
        if self.max_wall_minutes <= 0 or self.max_wall_minutes > MAX_RUN_MINUTES:
            raise ValueError("max_wall_minutes must obey MAX_RUN_MINUTES")

    @property
    def fingerprint(self) -> str:
        return content_sha(asdict(self))

    def manifest(self) -> ExperimentCampaignV1:
        fingerprint = self.fingerprint
        return ExperimentCampaignV1(
            campaign_id=self.campaign_id,
            experiment_id="abstract-cot-warmup",
            hypothesis="Repeated on-policy traces can be collected and handed to AP-019 with complete lineage.",
            decision="Whether the bounded warm-up loop is executable and resumable.",
            endpoints=(CampaignEndpointV1(endpoint_id="fixture_complete", metric="iterations", role="primary", direction="increase", minimum_effect=0.0),),
            arms=(CampaignArmV1(arm_id="warmup", role="candidate", config_sha256=fingerprint), CampaignArmV1(arm_id="bottleneck_only", role="control", config_sha256=fingerprint)),
            seeds=(self.seed,),
            budget=CampaignBudget(max_experiments=self.iterations, max_wall_minutes=self.max_wall_minutes),
            stopping_rules=("Stop on a failed iteration; never overwrite an incomplete iteration directory.",),
            controls=(CampaignControlV1(control_id="bottleneck_only", description="Collector disabled leaves the base training path untouched.", kind="negative"),),
            negative_controls=("bottleneck_only",),
            multiplicity_families=(MultiplicityFamilyV1(family_id="fixture", hypothesis_ids=("fixture_complete",), alpha=0.05),),
            promotion_gates=(CampaignGateV1(gate_id="fixture_complete", endpoint_id="fixture_complete", operator="ge", threshold=float(self.iterations)),),
            rollback_gates=(CampaignGateV1(gate_id="fixture_incomplete", endpoint_id="fixture_complete", operator="lt", threshold=float(self.iterations)),),
            artifact_requirements=(ArtifactRequirementV1(kind="version_stamp"),),
            claim_class=self.claim_class,
            source_commit=self.source_commit,
            source_dirty=False,
            author="codex",
            created_at="1970-01-01T00:00:00Z",
        )


@dataclass(frozen=True)
class IterationRecord:
    iteration: int
    parent_policy: str | None
    policy: str
    command: str
    config_sha256: str
    input_sha256: str
    output_sha256: str
    token_count: int
    update_count: int
    result_ref: str
    phases: dict[str, dict[str, Any]]
    reused: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WarmupResult:
    campaign_id: str
    manifest_sha256: str
    iterations: tuple[IterationRecord, ...]
    reused: int
    ran: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "manifest_sha256": self.manifest_sha256,
            "iterations": [row.to_dict() for row in self.iterations],
            "reused": self.reused,
            "ran": self.ran,
            "claim": "fixture wiring only; no locked evaluation, promotion, checkpoint, or ship claim",
        }


IterationRunner = Callable[[int, str | None, Path], Mapping[str, Any]]


def _completed(store: CampaignStore) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    events = store.root / "events.jsonl"
    if not events.exists():
        return rows
    for line in events.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event.get("event_type") == _EVENT:
            detail = event["detail"]
            rows[str(detail["input_sha256"])] = dict(detail["record"])
    return rows


def _validate_phases(phases: object) -> dict[str, dict[str, Any]]:
    if not isinstance(phases, Mapping) or set(phases) != _REQUIRED_PHASES:
        raise ValueError(f"iteration phases must be exactly {sorted(_REQUIRED_PHASES)}")
    validated: dict[str, dict[str, Any]] = {}
    for name, value in phases.items():
        if not isinstance(value, Mapping) or set(value) != _PHASE_FIELDS:
            raise ValueError(f"phase {name!r} must record exactly {sorted(_PHASE_FIELDS)}")
        if int(value["token_count"]) < 0:
            raise ValueError(f"phase {name!r} token_count must be non-negative")
        validated[str(name)] = {key: value[key] for key in _PHASE_FIELDS}
    return validated


def _record_from_runner(
    *, campaign: AbstractWarmupCampaignV1, iteration: int, parent: str | None, input_sha: str, raw: Mapping[str, Any]
) -> dict[str, Any]:
    required = {"policy", "command", "token_count", "update_count", "result_ref", "phases"}
    if missing := required - raw.keys():
        raise ValueError(f"iteration runner missing {sorted(missing)}")
    phases = _validate_phases(raw["phases"])
    if any(str(phase["config_sha256"]) != campaign.fingerprint for phase in phases.values()):
        raise ValueError("each phase must record the locked campaign configuration hash")
    return {
        "iteration": iteration,
        "parent_policy": parent,
        "policy": str(raw["policy"]),
        "command": str(raw["command"]),
        "config_sha256": campaign.fingerprint,
        "input_sha256": input_sha,
        "output_sha256": content_sha(raw),
        "token_count": int(raw["token_count"]),
        "update_count": int(raw["update_count"]),
        "result_ref": str(raw["result_ref"]),
        "phases": phases,
    }


def run_abstract_cot_warmup(
    campaign: AbstractWarmupCampaignV1, *, root: Path, runner: IterationRunner
) -> WarmupResult:
    """Run or reuse deterministic iterations without ever overwriting evidence.

    A completed event is the only valid resume point.  A directory without its
    event is deliberately left intact after interruption so the next invocation
    fails closed rather than duplicate collection or overwrite a policy artifact.
    """
    store = CampaignStore(campaign.campaign_id, root)
    store.initialize(CampaignSpec(campaign_id=campaign.campaign_id, objective="Bounded Abstract-CoT warm-up orchestration", primary_metric="fixture_complete", track="causal_lm", min_hypotheses=5, budget=CampaignBudget(max_experiments=campaign.iterations, max_wall_minutes=campaign.max_wall_minutes), created_at="1970-01-01T00:00:00Z"))
    lock = store.lock_experiment_campaign(campaign.manifest())
    completed = _completed(store)
    parent: str | None = None
    rows: list[IterationRecord] = []
    reused = ran = 0
    for iteration in range(1, campaign.iterations + 1):
        input_sha = content_sha({"campaign": campaign.fingerprint, "iteration": iteration, "parent_policy": parent})
        prior = completed.get(input_sha)
        was_reused = prior is not None
        if prior is None:
            iteration_root = store.root / "iterations" / f"{iteration:03d}"
            if iteration_root.exists():
                raise RuntimeError(
                    f"{iteration_root} exists without a completed ledger event; preserve it and inspect before retrying"
                )
            raw = runner(iteration, parent, iteration_root)
            prior = _record_from_runner(campaign=campaign, iteration=iteration, parent=parent, input_sha=input_sha, raw=raw)
            artifact = store.write_artifact("abstract_cot_warmup", prior)
            store.append_event(_EVENT, experiment_id="abstract-cot-warmup", status="wiring_only", artifact_sha256=artifact.stem, detail={"input_sha256": input_sha, "record": prior})
            ran += 1
        else:
            reused += 1
        row = IterationRecord(**prior, reused=was_reused)
        rows.append(row)
        parent = row.policy
    return WarmupResult(campaign.campaign_id, lock.manifest_sha256, tuple(rows), reused, ran)
