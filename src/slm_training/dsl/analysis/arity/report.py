"""Frozen data contracts + K^d capacity + report assembly (CAP0-02).

Everything a CAP0-02 arity certificate emits lives here as immutable, JSON-safe,
schema-versioned dataclasses:

* :class:`AnalysisBounds` — the declared bounded frame.
* :class:`StateSignature` — a hashable hard-state fingerprint (never logits,
  timestamps, or pids), used for the JSON round-trip contract and as the trie
  node identity.
* :class:`ExactArityReport` — the certified counts for one committed fixture,
  with deterministic :meth:`ExactArityReport.to_dict` / :meth:`from_dict`.

The capacity row uses pure integer arithmetic: ``min_k`` is the least ``K`` with
``K ** d >= minimized_state_count`` (no float ``log`` / ``pow`` rounding).

Honesty: the counts are certificates **for the committed fixture only**. They do
not reproduce the external CAP0-01 source-reported estimates (130 ASTs / 351 trie
/ 41 minimized / 162-190-345 / Hankel / residual); those remain source-reported
estimates per ``docs/design/calculated-arity-adaptive-precision.md`` and the raw
86-state value is retired from new conclusions.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

SCHEMA_VERSION = 1
SIGNATURE_VERSION = 1
# Bumped by hand if the reused ``production_codec`` action sigils change.
CODEC_VERSION = 1
# Bumped by hand if the arith-sketch parse -> canonical semantics change.
PARSER_VERSION = 1


class SchemaError(ValueError):
    """Raised when a serialized artifact carries a stale/unknown schema."""


@dataclass(frozen=True)
class AnalysisBounds:
    """Declared bounded frame for an arity analysis (immutable, JSON-safe)."""

    max_ast_nodes: int
    max_ast_depth: int | None = None
    max_live_bindings: int = 0
    template_classes: tuple[str, ...] = ()
    result_types: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.max_ast_nodes < 0:
            raise ValueError("max_ast_nodes must be non-negative")
        if self.max_ast_depth is not None and self.max_ast_depth < 0:
            raise ValueError("max_ast_depth must be non-negative")
        if self.max_live_bindings < 0:
            raise ValueError("max_live_bindings must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_ast_nodes": self.max_ast_nodes,
            "max_ast_depth": self.max_ast_depth,
            "max_live_bindings": self.max_live_bindings,
            "template_classes": list(self.template_classes),
            "result_types": list(self.result_types),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnalysisBounds:
        return cls(
            max_ast_nodes=int(data["max_ast_nodes"]),
            max_ast_depth=(
                None if data.get("max_ast_depth") is None
                else int(data["max_ast_depth"])
            ),
            max_live_bindings=int(data.get("max_live_bindings", 0)),
            template_classes=tuple(data.get("template_classes") or ()),
            result_types=tuple(data.get("result_types") or ()),
        )


@dataclass(frozen=True)
class StateSignature:
    """Hashable hard-state fingerprint of one derivation configuration.

    Only *hard* state is included (generation step, grammar prefix, frontier,
    scope window, expected type, template state) — never a logit, score,
    timestamp, or pid — so the fingerprint is stable and replayable.
    """

    version: int
    generation_order: int
    grammar_state: tuple[str, ...]
    frontier: tuple[str, ...]
    scope_signature: int
    expected_type: str
    template_state: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "generation_order": self.generation_order,
            "grammar_state": list(self.grammar_state),
            "frontier": list(self.frontier),
            "scope_signature": self.scope_signature,
            "expected_type": self.expected_type,
            "template_state": list(self.template_state),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StateSignature:
        version = int(data["version"])
        if version != SIGNATURE_VERSION:
            raise SchemaError(
                f"stale state-signature version {version} != {SIGNATURE_VERSION}"
            )
        return cls(
            version=version,
            generation_order=int(data["generation_order"]),
            grammar_state=tuple(data.get("grammar_state") or ()),
            frontier=tuple(data.get("frontier") or ()),
            scope_signature=int(data["scope_signature"]),
            expected_type=str(data["expected_type"]),
            template_state=tuple(data.get("template_state") or ()),
        )

    def fingerprint(self) -> str:
        """Deterministic sha256 over the canonical JSON of the hard state."""
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def min_alphabet_for_capacity(state_count: int, dimensions: int) -> int:
    """Least ``K`` with ``K ** dimensions >= state_count`` (integer arithmetic)."""
    if dimensions <= 0:
        raise ValueError("dimensions must be a positive integer")
    if state_count <= 1:
        return 1
    alphabet = 1
    while alphabet**dimensions < state_count:
        alphabet += 1
    return alphabet


@dataclass(frozen=True)
class ExactArityReport:
    """Certified arity counts for one committed bounded fixture."""

    schema_version: int
    fixture: str
    grammar_hash: str
    parser_version: int
    codec_version: int
    signature_version: int
    bounds: AnalysisBounds
    complete: bool
    canonical_ast_count: int
    raw_state_count: int
    trie_state_count: int
    minimized_state_count: int
    action_alphabet_size: int
    scope_signature_count: int
    branching_histogram: dict[int, int]
    max_local_branching: int
    forced_visit_fraction: dict[str, Any]
    completion_counts: dict[int, int]
    capacity: dict[str, int]
    work_counters: dict[str, int]
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "fixture": self.fixture,
            "grammar_hash": self.grammar_hash,
            "parser_version": self.parser_version,
            "codec_version": self.codec_version,
            "signature_version": self.signature_version,
            "bounds": self.bounds.to_dict(),
            "complete": self.complete,
            "canonical_ast_count": self.canonical_ast_count,
            "raw_state_count": self.raw_state_count,
            "trie_state_count": self.trie_state_count,
            "minimized_state_count": self.minimized_state_count,
            "action_alphabet_size": self.action_alphabet_size,
            "scope_signature_count": self.scope_signature_count,
            "branching_histogram": _int_key_map(self.branching_histogram),
            "max_local_branching": self.max_local_branching,
            "forced_visit_fraction": self.forced_visit_fraction,
            "completion_counts": _int_key_map(self.completion_counts),
            "capacity": dict(self.capacity),
            "work_counters": dict(self.work_counters),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExactArityReport:
        schema_version = int(data["schema_version"])
        if schema_version != SCHEMA_VERSION:
            raise SchemaError(
                f"stale report schema version {schema_version} != {SCHEMA_VERSION}"
            )
        for key in ("parser_version", "codec_version", "signature_version"):
            if key not in data:
                raise SchemaError(f"missing required version field: {key}")
        if int(data["signature_version"]) != SIGNATURE_VERSION:
            raise SchemaError("stale signature_version in report")
        return cls(
            schema_version=schema_version,
            fixture=str(data["fixture"]),
            grammar_hash=str(data["grammar_hash"]),
            parser_version=int(data["parser_version"]),
            codec_version=int(data["codec_version"]),
            signature_version=int(data["signature_version"]),
            bounds=AnalysisBounds.from_dict(data["bounds"]),
            complete=bool(data["complete"]),
            canonical_ast_count=int(data["canonical_ast_count"]),
            raw_state_count=int(data["raw_state_count"]),
            trie_state_count=int(data["trie_state_count"]),
            minimized_state_count=int(data["minimized_state_count"]),
            action_alphabet_size=int(data["action_alphabet_size"]),
            scope_signature_count=int(data["scope_signature_count"]),
            branching_histogram=_int_key_map(data["branching_histogram"]),
            max_local_branching=int(data["max_local_branching"]),
            forced_visit_fraction=dict(data["forced_visit_fraction"]),
            completion_counts=_int_key_map(data["completion_counts"]),
            capacity={k: int(v) for k, v in data["capacity"].items()},
            work_counters={k: int(v) for k, v in data["work_counters"].items()},
            provenance=dict(data.get("provenance") or {}),
        )

    def to_json(self) -> str:
        """Deterministic pretty JSON (stable digest across runs)."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def _int_key_map(mapping: dict[Any, Any]) -> dict[int, int]:
    """Normalise a histogram to ``{int: int}`` (JSON keys arrive as strings)."""
    return {int(key): int(value) for key, value in mapping.items()}


def analyze(
    *,
    fixture: str,
    bounds: AnalysisBounds,
    dimensions: int,
    max_programs: int = 1_000_000,
) -> ExactArityReport:
    """Run the exact pipeline and assemble the certified report.

    Enumerate -> canonicalise/dedupe -> validate-filter -> prefix trie ->
    acyclic minimisation -> branching / completion / capacity. Heavy imports are
    local so importing this module (and the CLI) stays cheap and Torch-free.
    """
    from slm_training.dsl.grammar.backends.types import GRAMMARS_DIR
    from slm_training.dsl.language_contract import contract_id
    from slm_training.dsl.analysis.arity.minimize import (
        branching_histogram,
        forced_visit_fraction,
        max_local_branching,
        minimal_completion_lengths,
        minimize,
    )
    from slm_training.dsl.analysis.arity.state_graph import (
        FIXTURES,
        EnumerationBounds,
        build_trie,
        enumerate_programs,
        scope_signature_values,
        structural_signatures,
    )

    if fixture not in FIXTURES:
        raise KeyError(f"unknown fixture {fixture!r}; known={sorted(FIXTURES)}")
    if dimensions <= 0:
        raise ValueError("dimensions must be a positive integer")
    spec = FIXTURES[fixture]
    operators = spec["operators"]

    enum_bounds = EnumerationBounds(
        max_ast_nodes=bounds.max_ast_nodes,
        max_ast_depth=bounds.max_ast_depth,
        max_live_bindings=bounds.max_live_bindings,
        operators=operators,
        template_classes=bounds.template_classes,
    )
    enumeration = enumerate_programs(enum_bounds, max_programs=max_programs)
    trie = build_trie(enumeration.programs, bounds.max_live_bindings)
    dfa = minimize(trie)

    canonical_ast_count = len(enumeration.programs)
    raw_state_count = len(structural_signatures(trie))
    trie_state_count = len(trie.nodes)
    minimized_state_count = dfa.class_count
    capacity = {
        "state_count": minimized_state_count,
        "d": dimensions,
        "min_k": min_alphabet_for_capacity(minimized_state_count, dimensions),
    }

    grammar_hash = hashlib.sha256(
        (GRAMMARS_DIR / "arith_sketch.lark").read_bytes()
    ).hexdigest()

    work_counters = {
        **enumeration.work,
        "canonical_ast_count": canonical_ast_count,
        "trie_nodes": trie_state_count,
        **dfa.work,
    }

    return ExactArityReport(
        schema_version=SCHEMA_VERSION,
        fixture=fixture,
        grammar_hash=grammar_hash,
        parser_version=PARSER_VERSION,
        codec_version=CODEC_VERSION,
        signature_version=SIGNATURE_VERSION,
        bounds=bounds,
        complete=enumeration.complete,
        canonical_ast_count=canonical_ast_count,
        raw_state_count=raw_state_count,
        trie_state_count=trie_state_count,
        minimized_state_count=minimized_state_count,
        action_alphabet_size=len(trie.action_alphabet()),
        scope_signature_count=len(scope_signature_values(trie)),
        branching_histogram=branching_histogram(dfa),
        max_local_branching=max_local_branching(dfa),
        forced_visit_fraction=forced_visit_fraction(dfa),
        completion_counts=minimal_completion_lengths(dfa),
        capacity=capacity,
        work_counters=work_counters,
        provenance={
            "backend": "arith-sketch",
            "grammar_file": "arith_sketch.lark",
            "operators": list(operators),
            "language_contract_id": contract_id(),
            "external_estimates_reproduced": False,
            "external_reference": (
                "docs/design/calculated-arity-adaptive-precision.md"
            ),
        },
    )
