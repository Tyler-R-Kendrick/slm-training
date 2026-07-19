"""Decision-difficulty contract for CAP1-02/03 grammar-state traces (SLM-173).

Computes a versioned, deterministic ``DecisionDifficulty`` record from a single
``GrammarDecisionTrace`` and aggregates per-program ``ProgramDifficulty`` from a
sequence of decision records. The features are intentionally narrow for the
first slice: arity, branch entropy, top-1 margin, and completion support.
Future slices can add confusability-graph quotient colors, binding depth, and
rare-action indicators without breaking the schema contract.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from slm_training.harnesses.distill.grammar_trace import GrammarDecisionTrace

SCHEMA_VERSION = "sde2-06.v1"


def _source_hash(seed: str | int, example_id: str, state_fingerprint: str) -> str:
    """Stable 16-hex hash of the trace provenance for source attribution."""
    payload = json.dumps(
        {"seed": str(seed), "example_id": example_id, "state_fingerprint": state_fingerprint},
        sort_keys=True,
        separators=(",", ":"),
    )
    return _digest(payload)[:16]


def _digest(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DecisionDifficulty:
    """Difficulty features for one grammar decision.

    All features are computed from compiler-legal action sets and the trace's
    own recorded distribution. No held-out outcome metric is used.
    """

    state_fingerprint: str
    live_legal_action_count: int
    log2_live_legal_action_count: float
    posterior_entropy_bits: float | None
    top1_margin: float | None
    completion_support_size_exact: int | None
    quotient_color: int | None = None
    source_hash: str | None = None
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "state_fingerprint": self.state_fingerprint,
            "live_legal_action_count": self.live_legal_action_count,
            "log2_live_legal_action_count": self.log2_live_legal_action_count,
            "posterior_entropy_bits": self.posterior_entropy_bits,
            "top1_margin": self.top1_margin,
            "completion_support_size_exact": self.completion_support_size_exact,
            "quotient_color": self.quotient_color,
            "source_hash": self.source_hash,
        }


@dataclass(frozen=True)
class ProgramDifficulty:
    """Aggregated difficulty for one program (a sequence of decisions)."""

    example_id: str
    decision_count: int
    mean_entropy_bits: float
    max_entropy_bits: float
    mean_arity: float
    max_arity: int
    source_hash: str | None = None
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "example_id": self.example_id,
            "decision_count": self.decision_count,
            "mean_entropy_bits": self.mean_entropy_bits,
            "max_entropy_bits": self.max_entropy_bits,
            "mean_arity": self.mean_arity,
            "max_arity": self.max_arity,
            "source_hash": self.source_hash,
        }


def decision_difficulty_from_trace(
    trace: GrammarDecisionTrace,
    *,
    quotient_color: int | None = None,
) -> DecisionDifficulty:
    """Extract a deterministic ``DecisionDifficulty`` from a grammar trace.

    Arity is the count of compiler-legal actions. Branch entropy is reused from
    the trace when logits/energies were captured; otherwise it is left as
    ``None`` to keep the record honest.
    """
    arity = len(trace.legal_action_ids)
    return DecisionDifficulty(
        state_fingerprint=trace.state_fingerprint,
        live_legal_action_count=arity,
        log2_live_legal_action_count=math.log2(max(arity, 1)),
        posterior_entropy_bits=trace.posterior_entropy_bits,
        top1_margin=trace.top1_margin,
        completion_support_size_exact=trace.completion_support_size_exact,
        quotient_color=quotient_color,
        source_hash=_source_hash(
            trace.seed, trace.example_id, trace.state_fingerprint
        ),
    )


def aggregate_program_difficulties(
    difficulties: list[DecisionDifficulty],
    *,
    example_id: str = "",
) -> ProgramDifficulty:
    """Aggregate a list of decision difficulties into a program summary.

    The aggregation is stable and deterministic; difficulties are not reordered.
    """
    if not difficulties:
        return ProgramDifficulty(
            example_id=example_id,
            decision_count=0,
            mean_entropy_bits=0.0,
            max_entropy_bits=0.0,
            mean_arity=0.0,
            max_arity=0,
        )

    entropies = [
        d.posterior_entropy_bits
        for d in difficulties
        if d.posterior_entropy_bits is not None
    ]
    arities = [d.live_legal_action_count for d in difficulties]
    source_hashes = sorted({d.source_hash for d in difficulties if d.source_hash})
    return ProgramDifficulty(
        example_id=example_id,
        decision_count=len(difficulties),
        mean_entropy_bits=sum(entropies) / len(entropies) if entropies else 0.0,
        max_entropy_bits=max(entropies) if entropies else 0.0,
        mean_arity=sum(arities) / len(arities),
        max_arity=max(arities),
        source_hash=_digest(
            json.dumps(source_hashes, separators=(",", ":"))
        )[:16]
        if source_hashes
        else None,
    )
