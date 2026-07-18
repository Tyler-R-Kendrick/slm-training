"""LDI3-04 immutable on-policy remine -> intervene -> regenerate campaign (SLM-132).

The OpenUI derivative of Auto-Antislop: generate on a frozen prompt set, profile and
localize structural failures, admit same-state action evidence, train one *removable*
intervention, regenerate under identical conditions, and measure repaired / persisted
/ regressed / newly-exposed / unresolved failure signatures across iterations.

This is an **autoresearch campaign integration, not a new pipeline and not a new
scheduler**. Immutability, content-addressed artifacts, and the hash-chained event log
are provided by the existing ``autoresearch.storage.CampaignStore``; content hashing by
``lineage.records.content_sha``. The four collaborators it orchestrates
(structural-slop forensics SLM-129, counterfactual action evidence SLM-131, structured
objectives SLM-128, and the removable adapter trainer SLM-126/122) sit behind narrow
backends so the model-generation / training surface stays out of the CPU smoke.

Scope of this issue (LDI3-04): the versioned fail-closed campaign config + per-stage
fingerprint, the stage DAG with content-addressed completion markers
(resume/immutability/duplicate-safe/invalidating), the iteration lifecycle with
admission-gated training, the failure-signature migration tables, the deterministic
stop rules, and a bounded **one-iteration fixture smoke** that publishes every expected
artifact with ``wiring only`` status. The frontier (model-backed) quality-bearing run
-- real generation, training, and five-suite/AgentV evaluation -- is deferred; it swaps
the fixture backends for real ones and updates the experiment matrix / model card then.
No RL, no automatic adapter composition, no learned code rewrite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from slm_training.autoresearch.schemas import CampaignBudget, CampaignSpec
from slm_training.autoresearch.storage import CampaignStore
from slm_training.lineage.records import content_sha

__all__ = [
    "CAMPAIGN_VERSION",
    "CAMPAIGN_TAG",
    "RemineConfigError",
    "RemineCampaignConfig",
    "GeneratedProgram",
    "GenerationBackend",
    "AdapterHandle",
    "TrainingBackend",
    "FixtureBackend",
    "FailureSignature",
    "MigrationReport",
    "migrate_signatures",
    "StopDecision",
    "evaluate_stop_rules",
    "IterationRecord",
    "CampaignResult",
    "STAGES_ITER0",
    "STAGES_ITERN",
    "campaign_spec_for",
    "run_campaign",
    "describe_campaign",
]

CAMPAIGN_VERSION = "ldi3-04-v1"
CAMPAIGN_TAG = "LDI-remine"

# Ordered stage DAG (each stage depends on the previous one within its iteration).
STAGES_ITER0 = (
    "evaluate_parent",
    "generate",
    "profile",
    "localize",
    "counterfactual",
    "decision_tables",
    "admission",
    "diagnostic",
)
STAGES_ITERN = (
    "train",
    "select",
    "regenerate",
    "profile",
    "verify",
    "migrate",
    "mine",
    "decide",
)

# Authorization outcomes recorded honestly after iteration 0 admission/diagnostic.
_AUTH = ("train_authorized", "repair_evidence", "no_safe_direction", "expired")
_ACTUATORS = ("twotower", "causal")


class RemineConfigError(ValueError):
    """Raised on an unknown config field or an out-of-contract value (fail closed)."""


@dataclass(frozen=True)
class RemineCampaignConfig:
    """Versioned campaign config. Everything the parent/intervention comparison must
    hold fixed is frozen here and copied (by fingerprint) into every iteration.
    Unknown fields fail closed via :meth:`from_mapping`."""

    campaign_id: str
    created_at: str  # deterministic; frozen at construction so resume is idempotent
    base_checkpoint_sha: str
    tokenizer_sha: str
    prompt_group_ids: tuple[str, ...]
    actuator_backend: str = "twotower"
    suite_mix: tuple[str, ...] = ()
    decode_config_hash: str = ""
    seeds: tuple[int, ...] = (0,)
    max_generation_tokens: int = 0
    verifier_bundle_hash: str = ""
    judge_bundle_hash: str = ""
    profile_config: dict[str, Any] = field(default_factory=dict)
    detector_config: dict[str, Any] = field(default_factory=dict)
    rollout_policy: dict[str, Any] = field(default_factory=dict)
    objective_config: dict[str, Any] = field(default_factory=dict)
    adapter_spec: dict[str, Any] = field(default_factory=dict)
    event_threshold: float = 0.0
    locality_threshold: float = 0.0
    end_to_end_threshold: float = 0.0
    min_new_evidence: int = 1
    max_iterations: int = 2
    max_wall_minutes: float = 5.0
    max_cost_units: float = 0.0
    max_tokens: int = 0
    notes: str = ""

    def __post_init__(self) -> None:
        if self.actuator_backend not in _ACTUATORS:
            raise RemineConfigError(f"actuator_backend must be one of {_ACTUATORS}")
        if self.max_iterations < 0 or self.max_iterations > 3:
            # Default max is two trained iterations; a third needs explicit justification.
            raise RemineConfigError("max_iterations must be in [0, 3]")
        if not self.prompt_group_ids:
            raise RemineConfigError("prompt_group_ids must be non-empty (frozen set)")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> RemineCampaignConfig:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = set(data) - known
        if unknown:
            raise RemineConfigError(f"unknown config field(s): {sorted(unknown)}")
        kw = dict(data)
        for seq in ("prompt_group_ids", "suite_mix", "seeds"):
            if seq in kw and kw[seq] is not None:
                kw[seq] = tuple(kw[seq])
        return cls(**kw)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            out[name] = list(value) if isinstance(value, tuple) else value
        return out

    def fingerprint(self) -> str:
        return content_sha(self.to_dict())


def campaign_spec_for(config: RemineCampaignConfig) -> CampaignSpec:
    """A valid autoresearch ``CampaignSpec`` for the store. ``created_at`` is taken from
    the (deterministic) config so ``CampaignStore.initialize`` is idempotent on resume."""
    return CampaignSpec(
        campaign_id=config.campaign_id,
        objective=(
            "Immutable on-policy remine/intervene/regenerate campaign that repairs "
            "structural-slop failures via one removable adapter (LDI3-04)."
        ),
        primary_metric="repaired_signature_rate",
        track="twotower" if config.actuator_backend == "twotower" else "causal_lm",
        budget=CampaignBudget(
            max_experiments=max(1, config.max_iterations + 2),
            max_wall_minutes=min(5.0, max(0.1, config.max_wall_minutes)),
        ),
        created_at=config.created_at,
        notes=f"{CAMPAIGN_TAG}:{CAMPAIGN_VERSION}",
    )


# --------------------------------------------------------------------------- #
# Backends (model generation / training). The CPU smoke uses FixtureBackend; the
# frontier run injects real backends that call the SLM-129/131/128/126 surfaces.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GeneratedProgram:
    """One traced on-policy generation."""

    program_id: str
    prompt_group: str
    corpus: str  # parent | intervention
    trace_id: str
    motifs: tuple[str, ...]  # structural-slop motif ids present
    failing_gate: str | None  # earliest failing verifier gate, or None if all pass

    def to_dict(self) -> dict[str, Any]:
        return {
            "program_id": self.program_id,
            "prompt_group": self.prompt_group,
            "corpus": self.corpus,
            "trace_id": self.trace_id,
            "motifs": list(self.motifs),
            "failing_gate": self.failing_gate,
        }


class GenerationBackend(Protocol):
    def generate(
        self, config: RemineCampaignConfig, *, corpus: str, adapter_id: str | None
    ) -> Sequence[GeneratedProgram]: ...


@dataclass(frozen=True)
class AdapterHandle:
    """A removable adapter with explicit parent lineage (never auto-merged)."""

    adapter_id: str
    base_checkpoint_sha: str
    parent_adapter_id: str | None
    spec_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter_id": self.adapter_id,
            "base_checkpoint_sha": self.base_checkpoint_sha,
            "parent_adapter_id": self.parent_adapter_id,
            "spec_fingerprint": self.spec_fingerprint,
        }


class TrainingBackend(Protocol):
    def train(
        self,
        config: RemineCampaignConfig,
        *,
        evidence: Mapping[str, Any],
        parent_adapter_id: str | None,
    ) -> AdapterHandle: ...


@dataclass(frozen=True)
class FixtureBackend:
    """Deterministic, torch-free generation + training for the wiring-only smoke.

    The parent corpus carries two motifs and two gate failures; a trained adapter
    repairs one motif and one gate while leaving the rest, and never fabricates a
    failure that was not present -- enough to exercise every migration category."""

    def generate(
        self, config: RemineCampaignConfig, *, corpus: str, adapter_id: str | None
    ) -> Sequence[GeneratedProgram]:
        programs: list[GeneratedProgram] = []
        repaired = adapter_id is not None
        for group in config.prompt_group_ids:
            base = content_sha({"g": group, "a": adapter_id, "c": corpus})[:8]
            # motif_a is repaired by the adapter; motif_b persists; gate g1 repaired, g2 persists.
            motifs = ("motif_b",) if repaired else ("motif_a", "motif_b")
            failing = "g2" if repaired else "g1"
            programs.append(
                GeneratedProgram(
                    program_id=f"{group}:{base}",
                    prompt_group=group,
                    corpus=corpus,
                    trace_id=f"trace:{base}",
                    motifs=motifs,
                    failing_gate=failing,
                )
            )
        return programs

    def train(
        self,
        config: RemineCampaignConfig,
        *,
        evidence: Mapping[str, Any],
        parent_adapter_id: str | None,
    ) -> AdapterHandle:
        spec_fp = content_sha({"spec": config.adapter_spec, "evidence": dict(evidence)})
        return AdapterHandle(
            adapter_id=f"adapter:{spec_fp[:12]}",
            base_checkpoint_sha=config.base_checkpoint_sha,
            parent_adapter_id=parent_adapter_id,  # explicit lineage; no auto-merge
            spec_fingerprint=spec_fp,
        )


# --------------------------------------------------------------------------- #
# Failure-signature migration
# --------------------------------------------------------------------------- #
_MIGRATION_CATEGORIES = ("repaired", "persisted", "regressed", "newly_exposed", "unresolved")


@dataclass(frozen=True)
class FailureSignature:
    """A named failure under a prompt group. ``supported`` is False when there was no
    admissible same-state evidence -- its disappearance can only be ``unresolved``,
    never ``repaired`` (aggregate disappearance from timeout/fallback is not repair)."""

    prompt_group: str
    kind: str  # motif | gate
    name: str
    supported: bool = True

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.prompt_group, self.kind, self.name)


@dataclass(frozen=True)
class MigrationReport:
    repaired: tuple[tuple[str, str, str], ...]
    persisted: tuple[tuple[str, str, str], ...]
    regressed: tuple[tuple[str, str, str], ...]
    newly_exposed: tuple[tuple[str, str, str], ...]
    unresolved: tuple[tuple[str, str, str], ...]

    def counts(self) -> dict[str, int]:
        return {c: len(getattr(self, c)) for c in _MIGRATION_CATEGORIES}

    @property
    def substitution_exceeds_improvement(self) -> bool:
        new = len(self.newly_exposed) + len(self.regressed)
        return new > len(self.repaired)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {c: [list(k) for k in getattr(self, c)] for c in _MIGRATION_CATEGORIES}
        out["counts"] = self.counts()
        out["substitution_exceeds_improvement"] = self.substitution_exceeds_improvement
        return out


def migrate_signatures(
    parent: Sequence[FailureSignature], child: Sequence[FailureSignature]
) -> MigrationReport:
    """Classify each failure signature across the parent->child regeneration.

    A signature present in the parent and absent in the child is ``repaired`` only if it
    was *supported* by admissible evidence; an unsupported disappearance is
    ``unresolved``. A newly-present signature is ``newly_exposed``. A signature present
    in both ``persisted``. ``regressed`` marks a previously-passing group's signature
    that appears with a strictly worse (later ``kind`` ordering) failure -- here modeled
    as a child-only ``gate`` failure on a group whose parent had only ``motif`` findings.
    """
    p_by_key = {s.key: s for s in parent}
    c_by_key = {s.key: s for s in child}
    p_groups_gate = {s.prompt_group for s in parent if s.kind == "gate"}
    repaired: list[tuple[str, str, str]] = []
    persisted: list[tuple[str, str, str]] = []
    regressed: list[tuple[str, str, str]] = []
    newly: list[tuple[str, str, str]] = []
    unresolved: list[tuple[str, str, str]] = []
    for key, sig in p_by_key.items():
        if key in c_by_key:
            persisted.append(key)
        elif sig.supported:
            repaired.append(key)
        else:
            unresolved.append(key)
    for key, sig in c_by_key.items():
        if key in p_by_key:
            continue
        if sig.kind == "gate" and sig.prompt_group not in p_groups_gate:
            regressed.append(key)
        else:
            newly.append(key)
    srt = lambda xs: tuple(sorted(xs))  # noqa: E731
    return MigrationReport(srt(repaired), srt(persisted), srt(regressed), srt(newly), srt(unresolved))


def signatures_from_programs(programs: Sequence[GeneratedProgram]) -> tuple[FailureSignature, ...]:
    sigs: list[FailureSignature] = []
    for p in programs:
        for motif in p.motifs:
            sigs.append(FailureSignature(p.prompt_group, "motif", motif))
        if p.failing_gate is not None:
            sigs.append(FailureSignature(p.prompt_group, "gate", p.failing_gate))
    return tuple(sigs)


# --------------------------------------------------------------------------- #
# Deterministic stop rules
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StopDecision:
    stop: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"stop": self.stop, "reason": self.reason}


@dataclass(frozen=True)
class CampaignState:
    """Inputs the stop rules read after an iteration (all deterministic)."""

    iteration: int
    max_iterations: int
    authorization: str
    protected_gate_regressed: bool
    end_to_end_improved: bool
    locality_within_budget: bool
    new_qualified_evidence: int
    min_new_evidence: int
    migration: MigrationReport | None
    replication_ok: bool = True
    budget_exhausted: bool = False


def evaluate_stop_rules(state: CampaignState) -> StopDecision:
    """Deterministic stop policy (order matters; first hit wins)."""
    if state.authorization == "no_safe_direction":
        return StopDecision(True, "no_safe_direction")
    if state.authorization == "expired" or state.budget_exhausted:
        return StopDecision(True, "budget_exhausted")
    if state.protected_gate_regressed:
        return StopDecision(True, "protected_gate_regressed")
    if not state.locality_within_budget:
        return StopDecision(True, "locality_or_latency_over_budget")
    if state.migration is not None and state.migration.substitution_exceeds_improvement:
        return StopDecision(True, "failure_substitution_exceeds_improvement")
    if state.new_qualified_evidence < state.min_new_evidence:
        return StopDecision(True, "new_evidence_below_threshold")
    if not state.end_to_end_improved:
        return StopDecision(True, "no_meaningful_end_to_end_improvement")
    if not state.replication_ok:
        return StopDecision(True, "positive_result_failed_replication")
    if state.iteration >= state.max_iterations:
        return StopDecision(True, "max_iterations_reached")
    return StopDecision(False, "continue")


# --------------------------------------------------------------------------- #
# Resume/immutability ledger over CampaignStore's content-addressed event log
# --------------------------------------------------------------------------- #
_STAGE_EVENT = "remine_stage_completed"
_STAGE_KIND = "remine_stage"


class _Ledger:
    """Layers content-addressed completion markers on ``CampaignStore``: each stage is
    keyed by (iteration, stage, input_fingerprint). A matching prior marker is reused
    (resume, duplicate-safe); a changed upstream fingerprint misses the marker and the
    stage re-runs (invalidation)."""

    def __init__(self, store: CampaignStore) -> None:
        self.store = store
        self._done: dict[tuple[int, str, str], str] = {}
        events = store.root / "events.jsonl"
        if events.exists():
            import json

            for line in events.read_text(encoding="utf-8").splitlines():
                if not line:
                    continue
                ev = json.loads(line)
                if ev.get("event_type") == _STAGE_EVENT:
                    d = ev.get("detail", {})
                    self._done[(d["iteration"], d["stage"], d["input_fp"])] = ev["artifact_sha256"]
        self.reused = 0
        self.ran = 0

    def run_or_reuse(self, iteration: int, stage: str, input_fp: str, fn) -> dict[str, Any]:
        key = (iteration, stage, input_fp)
        if key in self._done:
            self.reused += 1
            return self._read_artifact(self._done[key])
        output = fn()
        artifact = self.store.write_artifact(_STAGE_KIND, output)
        sha = artifact.stem
        self.store.append_event(
            _STAGE_EVENT,
            artifact_sha256=sha,
            status="wiring_only",
            detail={"iteration": iteration, "stage": stage, "input_fp": input_fp},
        )
        self._done[key] = sha
        self.ran += 1
        return output

    def _read_artifact(self, sha: str) -> dict[str, Any]:
        import json

        path = self.store.root / "artifacts" / _STAGE_KIND / f"{sha}.json"
        return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Iteration lifecycle + campaign loop
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class IterationRecord:
    iteration: int
    authorization: str
    adapter: AdapterHandle | None
    migration: MigrationReport | None
    stop: StopDecision
    stages: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "authorization": self.authorization,
            "adapter": self.adapter.to_dict() if self.adapter else None,
            "migration": self.migration.to_dict() if self.migration else None,
            "stop": self.stop.to_dict(),
            "stages": list(self.stages),
        }


@dataclass(frozen=True)
class CampaignResult:
    campaign_id: str
    version: str
    config_fingerprint: str
    iterations: tuple[IterationRecord, ...]
    status: str  # wiring_only (fixture) | frontier
    stages_run: int
    stages_reused: int

    def manifest(self) -> dict[str, Any]:
        """The deterministic, content-addressable campaign manifest. Excludes the
        run-vs-reuse execution counters so an interrupted+resumed run persists a
        byte-identical manifest to an uninterrupted one."""
        return {
            "campaign_id": self.campaign_id,
            "version": self.version,
            "tag": CAMPAIGN_TAG,
            "config_fingerprint": self.config_fingerprint,
            "status": self.status,
            "iterations": [it.to_dict() for it in self.iterations],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.manifest(),
            "stages_run": self.stages_run,
            "stages_reused": self.stages_reused,
        }


def _admission_authorization(config: RemineCampaignConfig, parent: Sequence[GeneratedProgram]) -> str:
    """Iteration-0 authorization. Training is authorized only when the frozen prompt set
    yielded a supported, admissible repair target. With no failing structural evidence
    there is no safe adapter direction."""
    has_failure = any(p.motifs or p.failing_gate for p in parent)
    if not config.prompt_group_ids:
        return "expired"
    return "train_authorized" if has_failure else "no_safe_direction"


def run_campaign(
    config: RemineCampaignConfig,
    *,
    backend: GenerationBackend | TrainingBackend | None = None,
    root: Path | str = Path("outputs/autoresearch"),
    status: str = "wiring_only",
) -> CampaignResult:
    """Run the immutable remine campaign. With the default ``FixtureBackend`` this is the
    bounded, torch-free one-iteration smoke that publishes every expected artifact and is
    resumable at every stage (re-running reuses completed markers)."""
    backend = backend or FixtureBackend()
    gen: GenerationBackend = backend  # type: ignore[assignment]
    trainer: TrainingBackend = backend  # type: ignore[assignment]

    store = CampaignStore(config.campaign_id, root=root)
    store.initialize(campaign_spec_for(config))
    store.write_artifact("remine_config", config.to_dict())
    ledger = _Ledger(store)
    cfp = config.fingerprint()

    def fp(*parts: Any) -> str:
        return content_sha([cfp, list(parts)])

    # -- Iteration 0: parent baseline + admission (no training unless authorized) --
    parent_programs: list[GeneratedProgram] = []
    for stage in STAGES_ITER0:
        if stage == "generate":
            out = ledger.run_or_reuse(0, stage, fp(stage), lambda: {
                "programs": [p.to_dict() for p in gen.generate(config, corpus="parent", adapter_id=None)]
            })
            parent_programs = [GeneratedProgram(**{**d, "motifs": tuple(d["motifs"])}) for d in out["programs"]]
        else:
            ledger.run_or_reuse(0, stage, fp(stage), lambda s=stage: {"stage": s, "status": status})
    authorization = _admission_authorization(config, parent_programs)
    parent_sigs = signatures_from_programs(parent_programs)

    iterations: list[IterationRecord] = [
        IterationRecord(
            iteration=0,
            authorization=authorization,
            adapter=None,
            migration=None,
            stop=evaluate_stop_rules(
                CampaignState(
                    iteration=0,
                    max_iterations=config.max_iterations,
                    authorization=authorization,
                    protected_gate_regressed=False,
                    end_to_end_improved=True,
                    locality_within_budget=True,
                    new_qualified_evidence=len(parent_sigs),
                    min_new_evidence=config.min_new_evidence,
                    migration=None,
                )
            ),
            stages=STAGES_ITER0,
        )
    ]

    # -- Iterations n>=1: one fresh removable adapter each, admission-gated --
    parent_adapter: str | None = None
    prev_sigs = parent_sigs
    n = 1
    while (
        authorization == "train_authorized"
        and not iterations[-1].stop.stop
        and n <= config.max_iterations
    ):
        adapter = trainer.train(config, evidence={"iteration": n, "config_fp": cfp}, parent_adapter_id=parent_adapter)
        child_programs: list[GeneratedProgram] = []
        for stage in STAGES_ITERN:
            if stage == "regenerate":
                out = ledger.run_or_reuse(n, stage, fp(n, stage, adapter.adapter_id), lambda: {
                    "adapter_id": adapter.adapter_id,
                    "programs": [
                        p.to_dict() for p in gen.generate(config, corpus="intervention", adapter_id=adapter.adapter_id)
                    ],
                })
                child_programs = [GeneratedProgram(**{**d, "motifs": tuple(d["motifs"])}) for d in out["programs"]]
            else:
                ledger.run_or_reuse(n, stage, fp(n, stage, adapter.adapter_id), lambda s=stage: {"stage": s, "status": status})
        child_sigs = signatures_from_programs(child_programs)
        migration = migrate_signatures(prev_sigs, child_sigs)
        ledger.run_or_reuse(n, "migration_report", fp(n, "migration", adapter.adapter_id), lambda: migration.to_dict())
        improved = len(migration.repaired) > 0
        stop = evaluate_stop_rules(
            CampaignState(
                iteration=n,
                max_iterations=config.max_iterations,
                authorization=authorization,
                protected_gate_regressed=False,
                end_to_end_improved=improved,
                locality_within_budget=True,
                new_qualified_evidence=len(migration.persisted) + len(migration.newly_exposed),
                min_new_evidence=config.min_new_evidence,
                migration=migration,
            )
        )
        iterations.append(
            IterationRecord(
                iteration=n,
                authorization=authorization,
                adapter=adapter,
                migration=migration,
                stop=stop,
                stages=STAGES_ITERN,
            )
        )
        parent_adapter = adapter.adapter_id  # explicit lineage; not merged
        prev_sigs = child_sigs
        n += 1

    result = CampaignResult(
        campaign_id=config.campaign_id,
        version=CAMPAIGN_VERSION,
        config_fingerprint=cfp,
        iterations=tuple(iterations),
        status=status,
        stages_run=ledger.ran,
        stages_reused=ledger.reused,
    )
    store.write_artifact("remine_campaign_manifest", result.manifest())
    store.append_event("remine_campaign_completed", status=status, detail={"iterations": len(iterations)})
    return result


def describe_campaign(config: RemineCampaignConfig) -> dict[str, Any]:
    """Resolve the config, stage DAG, arms, and identities without running anything."""
    return {
        "campaign_id": config.campaign_id,
        "version": CAMPAIGN_VERSION,
        "tag": CAMPAIGN_TAG,
        "config_fingerprint": config.fingerprint(),
        "actuator_backend": config.actuator_backend,
        "max_iterations": config.max_iterations,
        "stages_iter0": list(STAGES_ITER0),
        "stages_itern": list(STAGES_ITERN),
        "arms": [
            {"arm_id": "R0", "role": "parent baseline"},
            {"arm_id": "R1", "role": "one-shot intervention"},
            {"arm_id": "R2", "role": "second remine (only if R1 passes continuation)"},
            {"arm_id": "R1-no-remine", "role": "extra-training control"},
        ],
        "authorizations": list(_AUTH),
        "stop_reasons": [
            "no_safe_direction",
            "budget_exhausted",
            "protected_gate_regressed",
            "locality_or_latency_over_budget",
            "failure_substitution_exceeds_improvement",
            "new_evidence_below_threshold",
            "no_meaningful_end_to_end_improvement",
            "positive_result_failed_replication",
            "max_iterations_reached",
        ],
        "identities": {
            "base_checkpoint_sha": config.base_checkpoint_sha,
            "tokenizer_sha": config.tokenizer_sha,
            "decode_config_hash": config.decode_config_hash,
            "verifier_bundle_hash": config.verifier_bundle_hash,
        },
    }
