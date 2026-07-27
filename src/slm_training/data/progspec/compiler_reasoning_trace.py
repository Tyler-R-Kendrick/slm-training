"""LOT0-02 (SLM-249): ``CompilerReasoningTraceV1`` — the explicit compiler-grounded
reasoning trace that a future K x c looped-latent causal model would first learn
visibly and later internalize (LOT1, gated separately and not implemented here).

Scope (per the LOT0-01 / SLM-248 authorization's ``allowed_lot1_work``): define the
trace contract and its typed step schema, a lossless visible serialization, and
permutation-invariant / cycle-free / truncation-explicit helpers. No model code, no
learned plan predictor, and no corpus build live here -- this module only defines the
target and extracts it deterministically from an existing, already-validated
``ProgramSpec`` (see ``compiler_reasoning_trace_extract.py``).

Stage set justification (data-derived from repository evidence, not LOTUS analogy):
the six candidate stages from the issue map 1:1 onto concepts ``SemanticPlanV1``
(``slm_training.data.progspec.semantic_plan``) already extracts deterministically
from an OpenUI AST, plus one additional stage sourced from the production codec's
canonical statement order:

* ``intent_contract``            <- ``SemanticPlanV1.archetype``
* ``component_inventory``        <- ``SemanticPlanV1.role_slots``
* ``topology_skeleton``          <- ``SemanticPlanV1.topology`` (edges/siblings)
* ``semantic_edges_roles``       <- ``SemanticPlanV1.topology.parent_relation_candidates``
* ``binding_scope_references``   <- ``SemanticPlanV1.symbols`` / ``SemanticPlanV1.bindings``
* ``serialization_plan``         <- ``production_codec.statement_binding_order``

No stage is invented beyond what the repository's existing extractors already
compute; nothing here is a new compiler, parser, evaluator, or validity definition.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

TraceVersion = Literal["1"]

TraceStepKind = Literal[
    "intent_contract",
    "component_inventory",
    "topology_skeleton",
    "semantic_edges_roles",
    "binding_scope_references",
    "serialization_plan",
]

# Canonical stage order and the fixed K derived from repository evidence (see module
# docstring). Not inherited from LOTUS's K=6 by analogy -- it is coincidentally also
# six because that is exactly how many stages SemanticPlanV1 + the codec's statement
# order already distinguish.
CANONICAL_STAGE_ORDER: tuple[TraceStepKind, ...] = (
    "intent_contract",
    "component_inventory",
    "topology_skeleton",
    "semantic_edges_roles",
    "binding_scope_references",
    "serialization_plan",
)
K_MAX_STEPS = len(CANONICAL_STAGE_ORDER)

_DELIM_OPEN = re.compile(r"<step:([a-z_]+)>")
_DELIM_CLOSE = "</step>"
_TRACE_OPEN = "<trace_v1>"
_TRACE_CLOSE = "</trace_v1>"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TraceStepV1(_StrictModel):
    """One typed step of a ``CompilerReasoningTraceV1``.

    ``target_ids`` (target token IDs under a causal tokenizer/output geometry) is
    intentionally ``None`` in every step this module produces: reserving trace
    tokens in a shared vocabulary is model-implementation work gated behind a
    reviewed LOT1 contract (see ``docs/design/compiler-reasoning-trace-v1.md``),
    not authorized by this issue. Callers may populate it once that contract lands;
    the schema accepts it so no future migration is required.
    """

    trace_version: TraceVersion = "1"
    record_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    split: str = Field(min_length=1)
    fingerprint: str = Field(min_length=1)
    step_index: int = Field(ge=0)
    step_kind: TraceStepKind
    canonical_target: str
    accepted_targets: tuple[str, ...] = ()
    set_valued_fields: tuple[str, ...] = ()
    partial_order: tuple[TraceStepKind, ...] = ()
    support: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    ambiguity_reason: str | None = None
    is_truncated: bool = False
    source_refs: tuple[str, ...] = ()
    surface_tokens: tuple[str, ...] | None = None
    target_ids: tuple[int, ...] | None = None

    @model_validator(mode="after")
    def _check_accepted_targets_contain_canonical(self) -> "TraceStepV1":
        if self.accepted_targets and self.canonical_target not in self.accepted_targets:
            raise ValueError(
                "canonical_target must be a member of accepted_targets when "
                "accepted_targets is non-empty"
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TraceStepV1":
        return cls.model_validate(value)


class CompilerReasoningTraceV1(_StrictModel):
    """An ordered sequence of at most ``K_MAX_STEPS`` typed compiler-reasoning steps.

    ``steps`` are in root-identity order (``step_index`` 0..N-1); ``stage_order``
    names the accepted stage-kind order this trace declares, so a partial trace
    (curriculum stage < K) is representable by a strict prefix without violating
    ``partial_order`` cycle-freedom.
    """

    trace_version: TraceVersion = "1"
    record_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    split: str = Field(min_length=1)
    fingerprint: str = Field(min_length=1)
    stage_order: tuple[TraceStepKind, ...] = CANONICAL_STAGE_ORDER
    steps: tuple[TraceStepV1, ...] = ()

    @model_validator(mode="after")
    def _check_structure(self) -> "CompilerReasoningTraceV1":
        if len(self.stage_order) > K_MAX_STEPS:
            raise ValueError(
                f"stage_order declares {len(self.stage_order)} stages, exceeding "
                f"K_MAX_STEPS={K_MAX_STEPS}"
            )
        if len(set(self.stage_order)) != len(self.stage_order):
            raise ValueError("stage_order must not repeat a step_kind")
        stage_index = {kind: i for i, kind in enumerate(self.stage_order)}
        for expected_index, step in enumerate(self.steps):
            if step.step_index != expected_index:
                raise ValueError(
                    f"step_index {step.step_index} out of order at position "
                    f"{expected_index}"
                )
            if step.step_kind not in stage_index:
                raise ValueError(
                    f"step_kind {step.step_kind!r} is not declared in stage_order"
                )
            if step.record_id != self.record_id:
                raise ValueError("every step.record_id must match the trace record_id")
            if step.fingerprint != self.fingerprint:
                raise ValueError("every step.fingerprint must match the trace fingerprint")
        # Steps must appear in non-decreasing declared-stage order (a partial trace
        # is a strict prefix of stage_order, never out of order or duplicated).
        seen_kinds: list[TraceStepKind] = []
        last_stage_position = -1
        for step in self.steps:
            position = stage_index[step.step_kind]
            if position <= last_stage_position and step.step_kind in seen_kinds:
                raise ValueError(f"duplicate step_kind {step.step_kind!r} in trace")
            if position < last_stage_position:
                raise ValueError(
                    f"step_kind {step.step_kind!r} appears out of declared stage order"
                )
            last_stage_position = position
            seen_kinds.append(step.step_kind)
        validate_no_cycles(self.stage_order, self.steps)
        return self

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def is_partial(self) -> bool:
        """True when this trace covers a strict prefix of ``stage_order`` (curriculum stage < K)."""
        return self.step_count < len(self.stage_order)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CompilerReasoningTraceV1":
        return cls.model_validate(value)


def validate_no_cycles(
    stage_order: tuple[TraceStepKind, ...], steps: tuple[TraceStepV1, ...]
) -> None:
    """Fail closed if any step's ``partial_order`` dependency refs form a cycle.

    Dependencies are edges ``step_kind -> depends_on_kind``; a cycle means a stage
    is declared to depend (directly or transitively) on itself.
    """
    edges: dict[str, set[str]] = {kind: set() for kind in stage_order}
    declared_kinds = {step.step_kind for step in steps}
    for step in steps:
        for dep in step.partial_order:
            if dep not in declared_kinds and dep not in stage_order:
                raise ValueError(
                    f"step {step.step_kind!r} depends on undeclared stage {dep!r}"
                )
            edges.setdefault(step.step_kind, set()).add(dep)

    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {kind: WHITE for kind in edges}

    def visit(node: str, path: list[str]) -> None:
        color[node] = GRAY
        for neighbor in edges.get(node, ()):
            if color.get(neighbor, WHITE) == GRAY:
                cycle = " -> ".join((*path, neighbor))
                raise ValueError(f"cyclic partial_order dependency: {cycle}")
            if color.get(neighbor, WHITE) == WHITE:
                visit(neighbor, [*path, neighbor])
        color[node] = BLACK

    for node in list(edges):
        if color[node] == WHITE:
            visit(node, [node])


def accepted_set_equal(a: TraceStepV1, b: TraceStepV1) -> bool:
    """Permutation-invariant comparison of a step's accepted-alternative set.

    Two steps are accepted-set-equal when their ``step_kind`` and the *set* of
    accepted targets agree, regardless of declaration order (accepted alternatives
    are first-class, not ranked).
    """
    return a.step_kind == b.step_kind and set(a.accepted_targets or (a.canonical_target,)) == set(
        b.accepted_targets or (b.canonical_target,)
    )


def serialize_visible_trace(trace: CompilerReasoningTraceV1) -> str:
    """Typed-delimiter + canonical compact-token visible representation.

    Lossless (round-trips via :func:`parse_visible_trace`); distinct from the final
    OpenUI program surface; not optimized for human prose.
    """
    import json as _json

    lines = [_TRACE_OPEN]
    for step in trace.steps:
        payload = step.model_dump(mode="json", exclude={"step_kind"})
        compact = _json.dumps(payload, sort_keys=True, separators=(",", ":"))
        lines.append(f"<step:{step.step_kind}>{compact}{_DELIM_CLOSE}")
    lines.append(_TRACE_CLOSE)
    return "\n".join(lines)


def parse_visible_trace(
    text: str, *, stage_order: tuple[TraceStepKind, ...] = CANONICAL_STAGE_ORDER
) -> CompilerReasoningTraceV1:
    """Inverse of :func:`serialize_visible_trace`."""
    import json as _json

    body = text.strip()
    if not body.startswith(_TRACE_OPEN) or not body.endswith(_TRACE_CLOSE):
        raise ValueError("visible trace missing <trace_v1> ... </trace_v1> delimiters")
    inner = body[len(_TRACE_OPEN) : -len(_TRACE_CLOSE)].strip()
    steps: list[TraceStepV1] = []
    pos = 0
    while pos < len(inner):
        match = _DELIM_OPEN.search(inner, pos)
        if match is None:
            break
        step_kind = match.group(1)
        close_idx = inner.find(_DELIM_CLOSE, match.end())
        if close_idx == -1:
            raise ValueError(f"unterminated <step:{step_kind}> element")
        payload = _json.loads(inner[match.end() : close_idx])
        payload["step_kind"] = step_kind
        steps.append(TraceStepV1.model_validate(payload))
        pos = close_idx + len(_DELIM_CLOSE)
    if not steps:
        raise ValueError("visible trace contains no steps")
    first = steps[0]
    return CompilerReasoningTraceV1(
        record_id=first.record_id,
        source=first.source,
        split=first.split,
        fingerprint=first.fingerprint,
        stage_order=stage_order,
        steps=tuple(steps),
    )


def make_partial_trace(
    trace: CompilerReasoningTraceV1, stage_count: int
) -> CompilerReasoningTraceV1:
    """First ``stage_count`` stages only -- the LOT1 curriculum's partial-trace arm."""
    if not 0 <= stage_count <= len(trace.stage_order):
        raise ValueError(f"stage_count must be in [0, {len(trace.stage_order)}]")
    return trace.model_copy(update={"steps": trace.steps[:stage_count]})


def make_shuffled_control(
    trace: CompilerReasoningTraceV1, *, seed: int
) -> CompilerReasoningTraceV1:
    """Deterministic shuffled/wrong control: permute ``canonical_target`` values
    across this trace's own steps, preserving per-step length/support metadata
    while breaking the intended step->target correspondence.

    Never a positive corpus member -- diagnostic/oracle-arm use only (see
    ``docs/design/compiler-reasoning-trace-v1.md``, oracle ceiling experiment plan).
    """
    import random as _random

    if len(trace.steps) < 2:
        raise ValueError("shuffled control requires at least 2 steps to be a real permutation")
    rng = _random.Random(seed)
    targets = [step.canonical_target for step in trace.steps]
    permuted = targets[:]
    # Deterministic derangement retry loop, bounded and seed-driven.
    for _ in range(64):
        rng.shuffle(permuted)
        if all(a != b for a, b in zip(targets, permuted, strict=True)):
            break
    new_steps = tuple(
        step.model_copy(update={"canonical_target": new_target, "accepted_targets": ()})
        for step, new_target in zip(trace.steps, permuted, strict=True)
    )
    return trace.model_copy(update={"steps": new_steps})
