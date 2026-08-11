"""Explicit event traces and named machine cost models (KERN-06 / SLM-524).

Mirrors ``LeverProofLean.EventTrace``: abstract Nat costs over Event traces under
a parameterized ``MachineModel.cost``. Named models cover metrics already owned
by ``DecodeStats`` / train-loop phase timers. Decode projection is lossless for
represented counters; unobserved event kinds stay explicit unknowns — never
invented.

No API equates abstract cost with physical wall-clock without an explicit
``ThroughputAssumption``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

from slm_training.harness_core.lineage.records import content_sha
from slm_training.harnesses.reasoning.revmath.schemas import (
    AxisCertificateRefV1,
    RevmathFourAxisAnalysisV1,
)
from slm_training.models.decode_stats import DecodeStats, DecodeStatsRecordV1

EVENT_TRACE_SCHEMA = "event_trace_machine_cost/v1"
BOUND_AST_ID = "bound.event_trace.decode_unit_work.v1"
BOUND_AST_COMPOSE_ID = "bound.event_trace.compose.v1"
MACHINE_MODEL_REGISTRY_VERSION = 1


class EventKind(str, Enum):
    TOKENIZE = "tokenize"
    GRAMMAR_TRANSITION = "grammar_transition"
    SOLVER_EXPANSION = "solver_expansion"
    CERTIFICATE_CHECK = "certificate_check"
    NEURAL_FORWARD = "neural_forward"
    NEURAL_BACKWARD = "neural_backward"
    OPTIMIZER_STEP = "optimizer_step"
    MEMORY_READ = "memory_read"
    MEMORY_WRITE = "memory_write"
    KERNEL_LAUNCH = "kernel_launch"


# DecodeStats field each kind projects from (None ⇒ unobserved under decode).
DECODE_STATS_EVENT_OWNERS: dict[EventKind, str | None] = {
    EventKind.TOKENIZE: "tokens_emitted",
    EventKind.GRAMMAR_TRANSITION: "dfa_sync_count",
    EventKind.SOLVER_EXPANSION: "solver_expanded_nodes",
    EventKind.CERTIFICATE_CHECK: "solver_verifier_calls",
    EventKind.NEURAL_FORWARD: "forwards_count",
    EventKind.NEURAL_BACKWARD: None,  # train_loop timed("backward")
    EventKind.OPTIMIZER_STEP: None,  # train_loop timed("optim_step")
    EventKind.MEMORY_READ: "completion_full_prefix_lex_bytes",
    EventKind.MEMORY_WRITE: None,  # no decode write-byte counter
    EventKind.KERNEL_LAUNCH: None,  # no CUDA/host launch counter in DecodeStats
}

# Memory lex-bytes are optional witnesses; default decode unit-work omits them.
DECODE_UNIT_WORK_KINDS: tuple[EventKind, ...] = (
    EventKind.TOKENIZE,
    EventKind.GRAMMAR_TRANSITION,
    EventKind.SOLVER_EXPANSION,
    EventKind.CERTIFICATE_CHECK,
    EventKind.NEURAL_FORWARD,
)


@dataclass(frozen=True)
class Event:
    kind: EventKind
    payload: int

    def __post_init__(self) -> None:
        if self.payload < 0:
            raise ValueError("event payload must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind.value, "payload": self.payload}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Event:
        return cls(kind=EventKind(str(payload["kind"])), payload=int(payload["payload"]))


Trace = tuple[Event, ...]


@dataclass(frozen=True)
class MachineModel:
    """Parameterized cost model: ``cost(event) -> Nat`` plus identity/version."""

    id: str
    version: int
    cost_table: Mapping[EventKind, str]
    # cost_table values: "payload" (use event.payload) or "zero"

    def __post_init__(self) -> None:
        if self.version < 0:
            raise ValueError("model version must be non-negative")
        if not self.id:
            raise ValueError("model id required")
        missing = [k for k in EventKind if k not in self.cost_table]
        if missing:
            raise ValueError(f"cost_table missing kinds: {[m.value for m in missing]}")
        bad = {k.value: v for k, v in self.cost_table.items() if v not in ("payload", "zero")}
        if bad:
            raise ValueError(f"unsupported cost rules: {bad}")

    def cost(self, event: Event) -> int:
        rule = self.cost_table[event.kind]
        if rule == "zero":
            return 0
        return event.payload

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "version": self.version,
            "wall_clock": False,
            "cost_table": {k.value: v for k, v in self.cost_table.items()},
            "registry_version": MACHINE_MODEL_REGISTRY_VERSION,
        }


def _payload_kinds(*kinds: EventKind) -> dict[EventKind, str]:
    table = {k: "zero" for k in EventKind}
    for kind in kinds:
        table[kind] = "payload"
    return table


DECODE_UNIT_WORK_MODEL = MachineModel(
    id="DecodeUnitWorkModel",
    version=1,
    cost_table=_payload_kinds(*DECODE_UNIT_WORK_KINDS),
)

SOLVER_QUERY_MODEL = MachineModel(
    id="SolverQueryModel",
    version=1,
    cost_table=_payload_kinds(
        EventKind.SOLVER_EXPANSION, EventKind.CERTIFICATE_CHECK
    ),
)

NEURAL_FORWARD_ONLY_MODEL = MachineModel(
    id="NeuralForwardOnlyModel",
    version=1,
    cost_table=_payload_kinds(EventKind.NEURAL_FORWARD),
)

TRAIN_STEP_MODEL = MachineModel(
    id="TrainStepModel",
    version=1,
    cost_table=_payload_kinds(
        EventKind.NEURAL_FORWARD,
        EventKind.NEURAL_BACKWARD,
        EventKind.OPTIMIZER_STEP,
    ),
)

NAMED_MODELS: dict[str, MachineModel] = {
    DECODE_UNIT_WORK_MODEL.id: DECODE_UNIT_WORK_MODEL,
    SOLVER_QUERY_MODEL.id: SOLVER_QUERY_MODEL,
    NEURAL_FORWARD_ONLY_MODEL.id: NEURAL_FORWARD_ONLY_MODEL,
    TRAIN_STEP_MODEL.id: TRAIN_STEP_MODEL,
}


def trace_cost(model: MachineModel, trace: Sequence[Event]) -> int:
    """Fold ``model.cost`` over ``trace`` (Lean ``traceCost``)."""

    total = 0
    for event in trace:
        total += model.cost(event)
    return total


def trace_cost_append(
    model: MachineModel, left: Sequence[Event], right: Sequence[Event]
) -> int:
    """Composition law mirror: cost(left++right) = cost(left)+cost(right)."""

    return trace_cost(model, left) + trace_cost(model, right)


@dataclass(frozen=True)
class ThroughputAssumption:
    """External latency-per-cost-unit hypothesis — never inferred."""

    latency_per_cost_unit: int

    def __post_init__(self) -> None:
        if self.latency_per_cost_unit < 0:
            raise ValueError("latency_per_cost_unit must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "latency_per_cost_unit": self.latency_per_cost_unit,
            "external": True,
            "wall_clock_derived_in_kernel": False,
        }


def wall_clock_upper_bound(
    model: MachineModel, assumption: ThroughputAssumption, trace: Sequence[Event]
) -> int:
    """Abstract shape only; physical transfer requires the hypothesis."""

    return trace_cost(model, trace) * assumption.latency_per_cost_unit


def physical_latency_hypothesis_holds(
    model: MachineModel,
    assumption: ThroughputAssumption,
    trace: Sequence[Event],
    physical_ns: int,
) -> bool:
    """Mirror of Lean ``PhysicalLatencyHypothesis`` (caller-supplied check)."""

    if physical_ns < 0:
        raise ValueError("physical_ns must be non-negative")
    return physical_ns <= wall_clock_upper_bound(model, assumption, trace)


@dataclass(frozen=True)
class DecodeUnitWorkCounts:
    tokenize: int
    grammar_transition: int
    solver_expansion: int
    certificate_check: int
    neural_forward: int

    def __post_init__(self) -> None:
        for name in (
            "tokenize",
            "grammar_transition",
            "solver_expansion",
            "certificate_check",
            "neural_forward",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "tokenize": self.tokenize,
            "grammar_transition": self.grammar_transition,
            "solver_expansion": self.solver_expansion,
            "certificate_check": self.certificate_check,
            "neural_forward": self.neural_forward,
        }

    def as_trace(self) -> Trace:
        return (
            Event(EventKind.TOKENIZE, self.tokenize),
            Event(EventKind.GRAMMAR_TRANSITION, self.grammar_transition),
            Event(EventKind.SOLVER_EXPANSION, self.solver_expansion),
            Event(EventKind.CERTIFICATE_CHECK, self.certificate_check),
            Event(EventKind.NEURAL_FORWARD, self.neural_forward),
        )

    def unit_work_cost(self) -> int:
        return (
            self.tokenize
            + self.grammar_transition
            + self.solver_expansion
            + self.certificate_check
            + self.neural_forward
        )


def counts_from_stats_mapping(stats: Mapping[str, Any]) -> DecodeUnitWorkCounts:
    """Lossless extract of DecodeUnitWork axes from a DecodeStats ``as_dict``."""

    def _int(key: str) -> int:
        return int(stats.get(key, 0) or 0)

    return DecodeUnitWorkCounts(
        tokenize=_int("tokens_emitted"),
        grammar_transition=_int("dfa_sync_count"),
        solver_expansion=_int("solver_expanded_nodes"),
        certificate_check=_int("solver_verifier_calls"),
        neural_forward=_int("forwards_count"),
    )


def unobserved_event_kinds_for_decode() -> tuple[EventKind, ...]:
    return tuple(
        kind for kind, owner in DECODE_STATS_EVENT_OWNERS.items() if owner is None
    )


@dataclass(frozen=True)
class ProjectedDecodeTraceV1:
    """Projection of ``DecodeStatsRecordV1`` onto an abstract Event trace."""

    schema: str
    model: MachineModel
    counts: DecodeUnitWorkCounts
    trace: Trace
    abstract_cost: int
    unobserved_event_kinds: tuple[str, ...]
    projected_fields: dict[str, str]
    record_digest: str | None
    wall_clock_claim: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "machine_model": self.model.to_dict(),
            "counts": self.counts.to_dict(),
            "trace": [e.to_dict() for e in self.trace],
            "abstract_cost": self.abstract_cost,
            "unobserved_event_kinds": list(self.unobserved_event_kinds),
            "projected_fields": dict(self.projected_fields),
            "record_digest": self.record_digest,
            "wall_clock_claim": self.wall_clock_claim,
            "bound_ast_id": BOUND_AST_ID,
            "theorem_ref": "EventTrace.traceCost_append",
        }

    def certificate_sha256(self) -> str:
        return content_sha(self.to_dict())


def project_decode_stats(
    stats: DecodeStats | Mapping[str, Any],
    *,
    model: MachineModel = DECODE_UNIT_WORK_MODEL,
    record: DecodeStatsRecordV1 | None = None,
) -> ProjectedDecodeTraceV1:
    """Project decode counters without inventing missing event kinds."""

    mapping: Mapping[str, Any]
    if isinstance(stats, DecodeStats):
        mapping = stats.as_dict()
    else:
        mapping = stats
    counts = counts_from_stats_mapping(mapping)
    trace = counts.as_trace()
    cost = trace_cost(model, trace)
    if model.id == DECODE_UNIT_WORK_MODEL.id and cost != counts.unit_work_cost():
        raise AssertionError("DecodeUnitWorkModel cost diverged from count sum")
    projected = {
        kind.value: owner
        for kind, owner in DECODE_STATS_EVENT_OWNERS.items()
        if owner is not None and kind in DECODE_UNIT_WORK_KINDS
    }
    return ProjectedDecodeTraceV1(
        schema=EVENT_TRACE_SCHEMA,
        model=model,
        counts=counts,
        trace=trace,
        abstract_cost=cost,
        unobserved_event_kinds=tuple(k.value for k in unobserved_event_kinds_for_decode()),
        projected_fields=projected,
        record_digest=record.record_digest if record is not None else None,
        wall_clock_claim=False,
    )


def project_decode_stats_record(
    record: DecodeStatsRecordV1,
    *,
    model: MachineModel = DECODE_UNIT_WORK_MODEL,
) -> ProjectedDecodeTraceV1:
    return project_decode_stats(record.stats, model=model, record=record)


@dataclass(frozen=True)
class EventTraceEvidenceV1:
    projection: ProjectedDecodeTraceV1
    preconditions: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": EVENT_TRACE_SCHEMA,
            "projection": self.projection.to_dict(),
            "preconditions": list(self.preconditions),
            "wall_clock_claim": False,
            "bound_ast_id": BOUND_AST_ID,
            "resource_cost_model_id": (
                f"{self.projection.model.id}.v{self.projection.model.version}"
            ),
            "theorem_refs": (
                "EventTrace.traceCost_append",
                "EventTrace.cost_nonneg",
                "EventTrace.physical_bound_from_hypothesis",
            ),
        }

    def certificate_sha256(self) -> str:
        return content_sha(self.to_dict())


def build_event_trace_evidence(
    stats: DecodeStats | Mapping[str, Any] | DecodeStatsRecordV1,
    *,
    model: MachineModel = DECODE_UNIT_WORK_MODEL,
) -> EventTraceEvidenceV1:
    if isinstance(stats, DecodeStatsRecordV1):
        projection = project_decode_stats_record(stats, model=model)
    else:
        projection = project_decode_stats(stats, model=model)
    return EventTraceEvidenceV1(
        projection=projection,
        preconditions=(
            f"cost_model:{model.id}.v{model.version}",
            "abstract_nat_cost_not_wall_clock",
            "decode_projection_lossless_for_represented_counters",
            "unobserved_kinds_remain_unknown",
            "wall_clock_transfer_requires_explicit_throughput_assumption",
        ),
    )


def export_four_axis_event_trace_evidence(
    evidence: EventTraceEvidenceV1,
    *,
    formal_preflight_sha256: str | None = None,
) -> RevmathFourAxisAnalysisV1:
    """Persist model identity/version through the four-axis ledger."""

    cert = evidence.certificate_sha256()
    model = evidence.projection.model
    model_id = f"{model.id}.v{model.version}"
    return RevmathFourAxisAnalysisV1(
        assumption_strength=AxisCertificateRefV1(
            axis="assumption_strength",
            status="not_claimed",
            notes=(
                "KERN-06 cost laws are model-relative Nat folds; "
                "wall-clock needs an explicit ThroughputAssumption"
            ),
        ),
        computability=AxisCertificateRefV1(
            axis="computability",
            status="proved_axis",
            certificate_sha256=cert,
            theorem_ref="EventTrace.traceCost_append",
            notes="traceCost is additive under concatenation; costs are Nat",
        ),
        resource_bounds=AxisCertificateRefV1(
            axis="resource_bounds",
            status="proved_axis",
            certificate_sha256=cert,
            theorem_ref="EventTrace.decodeUnitWork_traceCost_eq_sum",
            bound_ast_id=BOUND_AST_ID,
            notes=(
                f"{model_id}: abstract decode unit-work cost; "
                "not a physical latency claim"
            ),
        ),
        implementation_refinement=AxisCertificateRefV1(
            axis="implementation_refinement",
            status="empirical_remainder",
            notes=(
                "Wall-clock ms fields on DecodeStats and unobserved "
                "backward/optimizer/kernel_launch/memory_write remain empirical"
            ),
        ),
        formal_preflight_sha256=formal_preflight_sha256,
    )


def resource_cost_model_id(model: MachineModel) -> str:
    """EVID-03 ledger field: named model identity + version."""

    return f"{model.id}.v{model.version}"


def frozen_fixture_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cases = (
        {
            "id": "unit_work_2_3_5_7_11",
            "counts": DecodeUnitWorkCounts(2, 3, 5, 7, 11),
            "expected_cost": 28,
        },
        {
            "id": "zero_work",
            "counts": DecodeUnitWorkCounts(0, 0, 0, 0, 0),
            "expected_cost": 0,
        },
        {
            "id": "forward_only_three",
            "counts": DecodeUnitWorkCounts(0, 0, 0, 0, 3),
            "expected_cost": 3,
        },
    )
    for case in cases:
        counts: DecodeUnitWorkCounts = case["counts"]  # type: ignore[assignment]
        cost = trace_cost(DECODE_UNIT_WORK_MODEL, counts.as_trace())
        rows.append(
            {
                "id": case["id"],
                "counts": counts.to_dict(),
                "abstract_cost": cost,
                "expected_cost": case["expected_cost"],
                "compose_left_right": trace_cost_append(
                    DECODE_UNIT_WORK_MODEL,
                    (Event(EventKind.NEURAL_FORWARD, 2),),
                    (Event(EventKind.SOLVER_EXPANSION, 3),),
                ),
                "machine_model": DECODE_UNIT_WORK_MODEL.to_dict(),
            }
        )
    return rows


__all__ = [
    "BOUND_AST_COMPOSE_ID",
    "BOUND_AST_ID",
    "DECODE_STATS_EVENT_OWNERS",
    "DECODE_UNIT_WORK_KINDS",
    "DECODE_UNIT_WORK_MODEL",
    "EVENT_TRACE_SCHEMA",
    "MACHINE_MODEL_REGISTRY_VERSION",
    "NAMED_MODELS",
    "NEURAL_FORWARD_ONLY_MODEL",
    "SOLVER_QUERY_MODEL",
    "TRAIN_STEP_MODEL",
    "DecodeUnitWorkCounts",
    "Event",
    "EventKind",
    "EventTraceEvidenceV1",
    "MachineModel",
    "ProjectedDecodeTraceV1",
    "ThroughputAssumption",
    "build_event_trace_evidence",
    "counts_from_stats_mapping",
    "export_four_axis_event_trace_evidence",
    "frozen_fixture_rows",
    "physical_latency_hypothesis_holds",
    "project_decode_stats",
    "project_decode_stats_record",
    "resource_cost_model_id",
    "trace_cost",
    "trace_cost_append",
    "unobserved_event_kinds_for_decode",
    "wall_clock_upper_bound",
]
