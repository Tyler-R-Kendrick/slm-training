"""Replay-only value-of-compute controller substrate (SIE-007 / SLM-473).

Two pieces, both diagnostic:

* :class:`ComputeStateFeaturesV1` -- a versioned, frozen decision-state feature
  vector read out of telemetry the repo already records
  (:class:`~slm_training.models.decode_stats.DecodeStatsRecordV1`). A feature
  the record does not carry is ``None`` (**UNKNOWN**) -- never 0.0, because a
  zero would silently read as "measured and empty" to a scoring rule.
* :class:`ControllerRuleV1` -- one bounded arithmetic scoring expression per
  candidate action, evaluated against those features to *recommend* where the
  next unit of compute should go.

The rule language is the symbolic-regression IR
(:mod:`slm_training.dsl.symbolic_expr_ir`) reused unchanged: its token grammar
admits only ``(``/``)``, allowlisted operators, and ``v<i>``/``c<i>`` symbol
references, so a rule cannot lex an arbitrary identifier, string, or numeric
literal -- there is no eval, no import, no call, and no reflection reachable
from a rule, and depth/node budgets are enforced at parse time. Comparisons and
booleans are expressed as score differences plus the abstain margin rather than
as a second operator set, so there is exactly one expression language in the
repo to fuzz and certify.

Authority (never negotiable here):

* The controller **cannot add or remove a legal candidate**. It returns an
  action recommendation and nothing else; :func:`controller_can_change_legality`
  is a hard ``False`` and is asserted in tests.
* An **exact singleton bypasses the controller entirely** (I2): a complete
  legal domain of size 1 commits deterministically with *zero* expression
  evaluations, never a soft preference from a learned score.
* Nothing here is enabled in production by this issue -- there is no
  registration into any serving path, only replay over frozen records.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from slm_training.dsl.symbolic_expr_ir import (
    SymbolicExprNode,
    collect_symbols,
    parse_expr,
    serialize_expr,
)
from slm_training.dsl.symbolic_expr_evaluator import evaluate_vectorized
from slm_training.models.decode_stats import DecodeStatsRecordV1

__all__ = [
    "COMPUTE_STATE_FEATURES_SCHEMA",
    "CONTROLLER_ACTIONS",
    "CONTROLLER_MAX_DEPTH",
    "CONTROLLER_MAX_NODES",
    "FEATURE_ORDER",
    "ComputeStateFeaturesV1",
    "ControllerRecommendationV1",
    "ControllerRuleV1",
    "ReplayReportV1",
    "controller_can_change_legality",
    "features_from_decode_record",
    "oracle_action_for_record",
    "replay_rule",
]

COMPUTE_STATE_FEATURES_SCHEMA = "compute_state_features/v1"

# Frozen feature vocabulary: index i binds `v<i>` in every controller rule, so
# a rule's meaning is pinned to this order. Appending a feature is a new schema
# version, never an in-place reorder.
FEATURE_ORDER: tuple[str, ...] = (
    "legal_domain_size",
    "legal_domain_complete",
    "ambiguous_rows_forwarded",
    "forwards_count",
    "denoiser_rows_evaluated",
    "forced_row_tokens_without_forward",
    "semantic_singleton_bypasses",
    "compiler_candidates",
    "probes_count",
    "attempts",
    "tokens_emitted",
    "unconstrained_retries",
    "prior_work_ms",
    "witness_count",
)

CONTROLLER_ACTIONS: tuple[str, ...] = (
    "deterministic_ranker",
    "neural_forward",
    "bounded_search",
    "repair",
    "abstain",
)

# Scoring rules are meant to stay readable and cheap; the SR defaults are far
# looser than a per-decision controller should ever need.
CONTROLLER_MAX_DEPTH = 8
CONTROLLER_MAX_NODES = 48

_SCORABLE_ACTIONS = tuple(a for a in CONTROLLER_ACTIONS if a != "abstain")


@dataclass(frozen=True)
class ComputeStateFeaturesV1:
    """Versioned decision-state features; ``None`` means UNKNOWN, not zero."""

    schema_version: str
    values: dict[str, float | None]
    legal_domain_status: str
    measurement_stage: str
    completeness: str

    def __post_init__(self) -> None:
        unknown_names = set(self.values) - set(FEATURE_ORDER)
        if unknown_names:
            raise ValueError(f"features outside the frozen vocabulary: {sorted(unknown_names)}")
        missing = set(FEATURE_ORDER) - set(self.values)
        if missing:
            raise ValueError(f"feature vector is not total; missing {sorted(missing)}")

    def feature_vector(self) -> tuple[float | None, ...]:
        return tuple(self.values[name] for name in FEATURE_ORDER)

    def unknown_features(self) -> tuple[str, ...]:
        return tuple(name for name in FEATURE_ORDER if self.values[name] is None)

    @property
    def is_exact_singleton(self) -> bool:
        """A *complete* legal domain of exactly one symbol. An incomplete or
        unknown domain of apparent size 1 is never a singleton (a budget or a
        pruning step could have produced it), so it does not bypass."""
        return (
            self.legal_domain_status == "complete"
            and self.values["legal_domain_size"] == 1.0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "values": dict(self.values),
            "legal_domain_status": self.legal_domain_status,
            "measurement_stage": self.measurement_stage,
            "completeness": self.completeness,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ComputeStateFeaturesV1":
        if payload.get("schema_version") != COMPUTE_STATE_FEATURES_SCHEMA:
            raise ValueError(
                f"unsupported compute state features schema: {payload.get('schema_version')!r}"
            )
        return cls(
            schema_version=payload["schema_version"],
            values=dict(payload["values"]),
            legal_domain_status=payload["legal_domain_status"],
            measurement_stage=payload["measurement_stage"],
            completeness=payload["completeness"],
        )


def _stat(stats: dict[str, Any], key: str) -> float | None:
    value = stats.get(key)
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def features_from_decode_record(record: DecodeStatsRecordV1) -> ComputeStateFeaturesV1:
    """Read features out of one frozen record. Only what the record actually
    carries becomes a number; everything else stays UNKNOWN."""
    stats = record.stats
    size = record.legal_domain_size
    status = record.legal_domain_status
    complete: float | None
    if status == "complete":
        complete = 1.0
    elif status in {"incomplete", "unsupported"}:
        complete = 0.0
    else:
        complete = None

    prior_ms = _stat(stats, "total_ms")
    values: dict[str, float | None] = {
        "legal_domain_size": None if size is None else float(size),
        "legal_domain_complete": complete,
        "ambiguous_rows_forwarded": _stat(stats, "ambiguous_rows_forwarded"),
        "forwards_count": _stat(stats, "forwards_count"),
        "denoiser_rows_evaluated": _stat(stats, "denoiser_rows_evaluated"),
        "forced_row_tokens_without_forward": _stat(stats, "forced_row_tokens_without_forward"),
        "semantic_singleton_bypasses": _stat(stats, "semantic_singleton_bypasses"),
        "compiler_candidates": _stat(stats, "compiler_candidates"),
        "probes_count": _stat(stats, "probes_count"),
        "attempts": _stat(stats, "attempts"),
        "tokens_emitted": _stat(stats, "tokens_emitted"),
        "unconstrained_retries": _stat(stats, "unconstrained_retries"),
        "prior_work_ms": prior_ms,
        "witness_count": float(len(record.witness_ids)),
    }
    return ComputeStateFeaturesV1(
        schema_version=COMPUTE_STATE_FEATURES_SCHEMA,
        values=values,
        legal_domain_status=status,
        measurement_stage=record.measurement_stage,
        completeness=record.completeness,
    )


@dataclass(frozen=True)
class ControllerRecommendationV1:
    """What the controller would advise, and how it got there."""

    action: str
    score: float | None
    reason: str
    expressions_evaluated: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "score": self.score,
            "reason": self.reason,
            "expressions_evaluated": self.expressions_evaluated,
        }


class ControllerRuleV1:
    """One bounded scoring expression per action, plus an abstain margin.

    Parsing happens once, here: an unparseable, over-budget, or
    out-of-vocabulary rule raises at construction rather than at replay time.
    """

    __slots__ = ("rule_id", "expressions", "coefficients", "abstain_margin")

    def __init__(
        self,
        *,
        rule_id: str,
        action_expressions: dict[str, str],
        coefficients: tuple[float, ...] = (),
        abstain_margin: float = 0.0,
    ) -> None:
        if not rule_id:
            raise ValueError("rule_id must be non-empty")
        if abstain_margin < 0.0:
            raise ValueError("abstain_margin must be >= 0")
        unknown = set(action_expressions) - set(_SCORABLE_ACTIONS)
        if unknown:
            raise ValueError(f"unscorable action(s): {sorted(unknown)}")
        if not action_expressions:
            raise ValueError("a rule must score at least one action")

        parsed: dict[str, SymbolicExprNode] = {}
        for action, text in action_expressions.items():
            node = parse_expr(
                text,
                num_variables=len(FEATURE_ORDER),
                max_depth=CONTROLLER_MAX_DEPTH,
                max_nodes=CONTROLLER_MAX_NODES,
            )
            _, coefficient_indices = collect_symbols(node)
            unbound = {i for i in coefficient_indices if i >= len(coefficients)}
            if unbound:
                raise ValueError(
                    f"action {action!r} references unbound coefficient(s) {sorted(unbound)}"
                )
            parsed[action] = node

        self.rule_id = rule_id
        self.expressions = parsed
        self.coefficients = coefficients
        self.abstain_margin = float(abstain_margin)

    def referenced_features(self) -> frozenset[str]:
        names: set[str] = set()
        for node in self.expressions.values():
            variables, _ = collect_symbols(node)
            names.update(FEATURE_ORDER[i] for i in variables)
        return frozenset(names)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "action_expressions": {
                action: serialize_expr(node) for action, node in sorted(self.expressions.items())
            },
            "coefficients": list(self.coefficients),
            "abstain_margin": self.abstain_margin,
        }

    def recommend(self, features: ComputeStateFeaturesV1) -> ControllerRecommendationV1:
        """Advise one action. Never consults a score when the grammar already
        decided (I2), and abstains rather than guessing on UNKNOWN input."""
        if features.is_exact_singleton:
            return ControllerRecommendationV1(
                action="deterministic_ranker",
                score=None,
                reason="exact singleton: committed without any controller evaluation",
                expressions_evaluated=0,
            )

        needed = self.referenced_features()
        unknown = sorted(needed & set(features.unknown_features()))
        if unknown:
            return ControllerRecommendationV1(
                action="abstain",
                score=None,
                reason=f"UNKNOWN feature(s) referenced by the rule: {unknown}",
                expressions_evaluated=0,
            )

        vector = [0.0 if v is None else v for v in features.feature_vector()]
        variables = np.asarray([vector], dtype=np.float64)
        coefficients = (
            np.asarray(self.coefficients, dtype=np.float64) if self.coefficients else None
        )

        scores: dict[str, float] = {}
        evaluated = 0
        for action, node in sorted(self.expressions.items()):
            result = evaluate_vectorized(node, variables=variables, coefficients=coefficients)
            evaluated += 1
            if not result.valid_mask[0]:
                return ControllerRecommendationV1(
                    action="abstain",
                    score=None,
                    reason=f"numeric-domain violation scoring {action!r}",
                    expressions_evaluated=evaluated,
                )
            scores[action] = float(result.predictions[0])

        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        best_action, best_score = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else None
        if runner_up is not None and (best_score - runner_up) < self.abstain_margin:
            return ControllerRecommendationV1(
                action="abstain",
                score=best_score,
                reason=(
                    f"margin {best_score - runner_up:.6g} below abstain_margin "
                    f"{self.abstain_margin:.6g}"
                ),
                expressions_evaluated=evaluated,
            )
        return ControllerRecommendationV1(
            action=best_action,
            score=best_score,
            reason="highest scoring action",
            expressions_evaluated=evaluated,
        )


def controller_can_change_legality() -> bool:
    """Always ``False``. The controller recommends where compute goes; the
    legal domain is the grammar's, and no rule may widen or narrow it."""
    return False


def oracle_action_for_record(record: DecodeStatsRecordV1) -> str:
    """The action the observed outcome says was actually right, in priority
    order. Derived only from counters the record already carries -- this is a
    read of history, never a label supplied by a rule under test."""
    features = features_from_decode_record(record)
    if features.is_exact_singleton:
        return "deterministic_ranker"
    stats = record.stats
    if float(stats.get("unconstrained_retries") or 0) > 0:
        return "repair"
    if float(stats.get("attempts") or 1) > 1:
        return "bounded_search"
    if float(stats.get("forwards_count") or 0) > 0:
        return "neural_forward"
    return "deterministic_ranker"


@dataclass(frozen=True)
class ReplayReportV1:
    """Per-record recommendations plus regret against the observed oracle."""

    rule_id: str
    recommendations: tuple[ControllerRecommendationV1, ...]
    oracle_actions: tuple[str, ...]
    agreements: int
    abstentions: int
    regret: float
    expressions_evaluated: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "recommendations": [r.to_dict() for r in self.recommendations],
            "oracle_actions": list(self.oracle_actions),
            "agreements": self.agreements,
            "abstentions": self.abstentions,
            "regret": self.regret,
            "expressions_evaluated": self.expressions_evaluated,
        }


def replay_rule(
    rule: ControllerRuleV1, records: tuple[DecodeStatsRecordV1, ...]
) -> ReplayReportV1:
    """Score a candidate rule against frozen records. Deterministic: the same
    rule and the same records always produce the same report."""
    recommendations: list[ControllerRecommendationV1] = []
    oracles: list[str] = []
    agreements = 0
    abstentions = 0
    evaluated = 0
    for record in records:
        features = features_from_decode_record(record)
        recommendation = rule.recommend(features)
        oracle = oracle_action_for_record(record)
        recommendations.append(recommendation)
        oracles.append(oracle)
        evaluated += recommendation.expressions_evaluated
        if recommendation.action == "abstain":
            abstentions += 1
        elif recommendation.action == oracle:
            agreements += 1
    decided = len(records) - abstentions
    # Abstention is scored as coverage loss rather than as a wrong answer, but
    # a rule that never decides has no measured skill at all -- so it takes
    # maximal regret instead of the vacuous 1 - 0/0 = 0 a do-nothing rule would
    # otherwise use to outrank every rule that actually committed.
    regret = 1.0 if decided == 0 else 1.0 - (agreements / decided)
    return ReplayReportV1(
        rule_id=rule.rule_id,
        recommendations=tuple(recommendations),
        oracle_actions=tuple(oracles),
        agreements=agreements,
        abstentions=abstentions,
        regret=regret,
        expressions_evaluated=evaluated,
    )