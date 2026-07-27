"""Real (non-synthetic-score-injected) corpus for TurnDispositionHead -- VAR3-03 (SLM-433).

VAR3-02 (SLM-430) shipped ``TurnDispositionLabel``/``turn_disposition_losses``/
``TurnDispositionReportV1`` (``models/turn_disposition_head.py``) but tested
them only against synthetic tensors -- its own design doc named a future
corpus-labeling issue as the thing that would have to build real supervision.
This module is that corpus builder.

Every trace is a real, fresh-enumerated conversation over a small fixture
``OperatorLibraryV1`` (the same real DSL pack/operator/legal-set machinery
``data.flow.operator_policy_corpus`` already uses), not a hand-scored
synthetic tensor:

- ``out_of_scope`` / ``answer`` / forced-singleton ``emit`` labels are read
  directly off a freshly re-enumerated ``OperatorLegalSetV1`` (never a model
  prediction), exactly mirroring ``dsl.operators.turn_disposition
  .derive_turn_disposition``'s own I1 precedence -- this module must never
  assign ``out_of_scope`` to a turn whose legal set is non-empty, and a test
  proves it.
- ``clarify``-vs-``emit`` margin examples come from a real per-trace choice
  between two live legal actions with genuinely different *declared*
  operator costs (``AstOperatorV1.cost`` -- a real, structural field every
  operator in this repo already carries), never an injected/perturbed score.
  Each trace's real ground-truth accepted action is drawn from a per-tier
  Bernoulli rate that increases with the cost margin -- a large real cost gap
  usually (not always) is the intended action; a near-tie is close to a
  coinflip. This is the real, held-out-testable signal a trained classifier
  can either recover or fail to recover; nothing about an individual
  example's score is injected to force a margin.

Split is leakage-safe at the *trace* level (an entire trace's four rows stay
in one split), never per-row, since all four rows of a trace share the same
underlying state chain.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, replace
from typing import Any

from slm_training.dsl.operators.contracts import (
    ActionEffectV1,
    ApplicationProvenanceV1,
    AstOperatorV1,
    CompilerCoverage,
    _fingerprint,
)
from slm_training.dsl.operators.legal_set import (
    LegalSetCoverage,
    OperatorLegalSetV1,
    enumerate_operator_legal_set,
)
from slm_training.dsl.operators.references import build_reference_table
from slm_training.dsl.operators.registry import (
    OperatorLibraryV1,
    OperatorMutationV1,
    OperatorRejectedError,
    OperatorStateV1,
    RegisteredOperatorV1,
)
from slm_training.dsl.pack import get_pack
from slm_training.models.turn_disposition_head import (
    TurnDispositionLabel,
    TurnDispositionTargetV1,
)

__all__ = [
    "TIER_MARGINS",
    "TIER_GOLD_CHEAP_PROBABILITY",
    "TurnDispositionCorpusRowV1",
    "TurnDispositionCorpusQualityReportV1",
    "TurnDispositionCorpusV1",
    "build_turn_disposition_corpus",
]

# --------------------------------------------------------------------------- #
# Preregistered tier design (locked before any corpus is drawn).
#
# Five real declared-cost margins between two live legal actions ("cheap" /
# "exp"); ``TIER_GOLD_CHEAP_PROBABILITY`` is the true per-tier rate at which
# the real ground-truth accepted action is the cheaper one -- increasing with
# margin, as a real cost gap usually (not always) reflects the intended
# action. Both tuples are fixture design constants, not per-example noise.
# --------------------------------------------------------------------------- #
TIER_MARGINS: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 1.0)
TIER_GOLD_CHEAP_PROBABILITY: tuple[float, ...] = (0.5, 0.6, 0.75, 0.85, 0.95)
CHEAP_COST: float = 1.0

_PACK_ID = "openui"
_START_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(rf'^root = TextContent\(":t(\d+)_pair{tier}\.start"\)$')
    for tier in range(len(TIER_MARGINS))
)
_P_PATTERN = re.compile(r'^root = TextContent\(":t(\d+)\.p"\)$')
_MID_PATTERN = re.compile(r'^root = TextContent\(":t(\d+)\.mid"\)$')


def _declaration(operator_id: str, *, cost: float) -> AstOperatorV1:
    return AstOperatorV1(
        operator_id=operator_id,
        version="v1",
        domain="openui.ast",
        codomain="openui.ast",
        argument_slots=(),
        preconditions=(),
        effect_signature=(),
        locality="node",
        cost=cost,
    )


def _stage_executor(pattern: re.Pattern[str], next_suffix: str):
    def _execute(state: OperatorStateV1, _arguments: tuple) -> OperatorMutationV1:
        match = pattern.fullmatch(state.source)
        if match is None:
            raise OperatorRejectedError("slm433_disposition.precondition_failed")
        trace_ref = match.group(1)
        return OperatorMutationV1(
            source=f'root = TextContent(":t{trace_ref}.{next_suffix}")',
            effect=ActionEffectV1(compiler_coverage=CompilerCoverage.EXACT),
        )

    return _execute


def _build_library() -> OperatorLibraryV1:
    """A small, fixed, real operator library exercising every I1 branch.

    Per trace ``n`` (tier ``k`` baked into the root placeholder text):
    ``start`` (2 live candidates, real cost margin) -> ``p`` (forced
    singleton) -> ``mid`` (forced singleton, but externally flagged
    ``is_state_query`` so I1 resolves it as ``answer``) -> ``final`` (no
    operator's precondition matches -- genuinely empty legal set).
    """
    registered: list[RegisteredOperatorV1] = []
    for tier, margin in enumerate(TIER_MARGINS):
        registered.append(
            RegisteredOperatorV1(
                _declaration(f"disposition.cheap_tier{tier}", cost=CHEAP_COST),
                _stage_executor(_START_PATTERNS[tier], "p"),
            )
        )
        registered.append(
            RegisteredOperatorV1(
                _declaration(
                    f"disposition.exp_tier{tier}", cost=CHEAP_COST + margin
                ),
                _stage_executor(_START_PATTERNS[tier], "p"),
            )
        )
    registered.append(
        RegisteredOperatorV1(
            _declaration("disposition.advance_p_to_mid", cost=1.0),
            _stage_executor(_P_PATTERN, "mid"),
        )
    )
    registered.append(
        RegisteredOperatorV1(
            _declaration("disposition.advance_mid_to_final", cost=1.0),
            _stage_executor(_MID_PATTERN, "final"),
        )
    )
    return OperatorLibraryV1(tuple(registered))


@dataclass(frozen=True)
class TurnDispositionCorpusRowV1:
    """One real, freshly-enumerated turn. ``target``/``features`` are only
    populated for ``turn_kind == "scored"`` -- the only stage at which I1
    precedence ever reaches a learned decision; every other kind is resolved
    by a derived rule the classifier never sees."""

    row_id: str
    trace_id: str
    tier: int
    turn_index: int
    turn_kind: str  # "scored" | "forced_emit" | "answer" | "out_of_scope"
    is_state_query: bool
    legal_action_count: int
    coverage_complete: bool
    forced_singleton: bool
    gold_pick: str | None  # "cheap" | "exp" | None
    features: tuple[float, ...]  # (margin, cheap_cost, exp_cost) or ()
    target: TurnDispositionTargetV1 | None
    split: str
    schema: str = "turn_disposition_corpus_row/v1"

    def __post_init__(self) -> None:
        if self.turn_kind == "out_of_scope" and self.legal_action_count != 0:
            raise ValueError(
                "out_of_scope rows must be provably derived from an empty "
                "legal set, never assigned to a non-empty one"
            )
        if self.turn_kind != "out_of_scope" and self.legal_action_count == 0:
            raise ValueError(
                f"turn_kind={self.turn_kind!r} requires a non-empty legal set"
            )
        if self.turn_kind == "scored" and (self.target is None or not self.features):
            raise ValueError("scored rows require both target and features")
        if self.split not in {"train", "heldout"}:
            raise ValueError("unsupported split")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "row_id": self.row_id,
            "trace_id": self.trace_id,
            "tier": self.tier,
            "turn_index": self.turn_index,
            "turn_kind": self.turn_kind,
            "is_state_query": self.is_state_query,
            "legal_action_count": self.legal_action_count,
            "coverage_complete": self.coverage_complete,
            "forced_singleton": self.forced_singleton,
            "gold_pick": self.gold_pick,
            "features": list(self.features),
            "target": (
                {"label": self.target.label.value, "forced_singleton": self.target.forced_singleton}
                if self.target is not None
                else None
            ),
            "split": self.split,
        }


@dataclass(frozen=True)
class TurnDispositionCorpusQualityReportV1:
    """Aggregate denominators for the VAR3-03 corpus build."""

    total_traces: int
    total_rows: int
    rows_by_kind: dict[str, int]
    rows_by_split: dict[str, int]
    rows_by_tier: dict[int, int]
    out_of_scope_never_from_nonempty_set: bool
    schema: str = "turn_disposition_corpus_quality_report/v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "total_traces": self.total_traces,
            "total_rows": self.total_rows,
            "rows_by_kind": dict(sorted(self.rows_by_kind.items())),
            "rows_by_split": dict(sorted(self.rows_by_split.items())),
            "rows_by_tier": {str(k): v for k, v in sorted(self.rows_by_tier.items())},
            "out_of_scope_never_from_nonempty_set": (
                self.out_of_scope_never_from_nonempty_set
            ),
        }


@dataclass(frozen=True)
class TurnDispositionCorpusV1:
    rows: tuple[TurnDispositionCorpusRowV1, ...]
    quality_report: TurnDispositionCorpusQualityReportV1
    schema: str = "turn_disposition_corpus/v1"

    def by_split(self, split: str) -> tuple[TurnDispositionCorpusRowV1, ...]:
        return tuple(row for row in self.rows if row.split == split)

    def scored(self, split: str | None = None) -> tuple[TurnDispositionCorpusRowV1, ...]:
        rows = self.rows if split is None else self.by_split(split)
        return tuple(row for row in rows if row.turn_kind == "scored")


def _state_and_legal_set(
    *, pack, library: OperatorLibraryV1, source: str, request_id: str, seed: int
) -> tuple[OperatorStateV1, OperatorLegalSetV1]:
    state = OperatorStateV1.from_source(pack, source)
    branch = _fingerprint({"schema": "slm433_branch/v1", "request_id": request_id})
    table = build_reference_table(
        request_id=request_id,
        state_digest=state.state_digest,
        branch_digest=branch,
        descriptors=(),
        seed=seed,
    )
    provenance = ApplicationProvenanceV1(
        pack_id=pack.pack_id,
        compiler_id="slm433.disposition_fixture",
        compiler_version="v1",
        source_artifact_digest=_fingerprint({"source": source}),
        request_id=request_id,
    )
    legal_set = enumerate_operator_legal_set(
        pack=pack,
        library=library,
        state=state,
        reference_table=table,
        provenance=provenance,
    )
    return state, legal_set


def _split_trace_ids(
    trace_ids_by_tier: dict[int, list[str]], *, train_fraction: float, seed: int
) -> dict[str, str]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    rng = random.Random(seed)
    split_by_trace: dict[str, str] = {}
    for tier, trace_ids in sorted(trace_ids_by_tier.items()):
        ordered = sorted(trace_ids)
        rng.shuffle(ordered)
        n_train = max(1, min(len(ordered) - 1, round(len(ordered) * train_fraction)))
        for trace_id in ordered[:n_train]:
            split_by_trace[trace_id] = "train"
        for trace_id in ordered[n_train:]:
            split_by_trace[trace_id] = "heldout"
    return split_by_trace


def build_turn_disposition_corpus(
    *, n_per_tier: int = 40, train_fraction: float = 0.7, seed: int = 433
) -> TurnDispositionCorpusV1:
    """Build the real, trace-leakage-safe VAR3-03 corpus.

    Four rows per trace: ``scored`` (real cost-margin clarify/emit
    decision) -> ``forced_emit`` (singleton, non-query) -> ``answer``
    (singleton legal set, but externally flagged ``is_state_query`` -- I1
    precedence resolves it as ``answer`` ahead of the forced-emit bypass,
    proving precedence order is preserved) -> ``out_of_scope`` (genuinely
    empty legal set).
    """
    if n_per_tier < 2:
        raise ValueError("n_per_tier must be at least 2 (one per split)")
    base_pack = get_pack(_PACK_ID)
    library = _build_library()
    pack = replace(base_pack, operator_library=library)

    trace_ids_by_tier: dict[int, list[str]] = {}
    gold_rng = random.Random(seed)
    plan: list[tuple[str, int]] = []  # (trace_id, tier)
    trace_counter = 0
    for tier in range(len(TIER_MARGINS)):
        for _ in range(n_per_tier):
            trace_id = f"trace{trace_counter:04d}"
            trace_counter += 1
            trace_ids_by_tier.setdefault(tier, []).append(trace_id)
            plan.append((trace_id, tier))

    split_by_trace = _split_trace_ids(
        trace_ids_by_tier, train_fraction=train_fraction, seed=seed
    )

    rows: list[TurnDispositionCorpusRowV1] = []
    for trace_id, tier in plan:
        margin = TIER_MARGINS[tier]
        n = trace_id.replace("trace", "")
        request_id = f"req-{trace_id}"
        split = split_by_trace[trace_id]

        # --- turn 0: start (scored clarify/emit) ---
        start_source = f'root = TextContent(":t{n}_pair{tier}.start")'
        _state0, legal0 = _state_and_legal_set(
            pack=pack, library=library, source=start_source, request_id=request_id, seed=1
        )
        cheap_cost = CHEAP_COST
        exp_cost = CHEAP_COST + margin
        gold_pick = (
            "cheap"
            if gold_rng.random() < TIER_GOLD_CHEAP_PROBABILITY[tier]
            else "exp"
        )
        emit_would_be_correct = gold_pick == "cheap"
        target = TurnDispositionTargetV1(
            label=(
                TurnDispositionLabel.EMIT
                if emit_would_be_correct
                else TurnDispositionLabel.CLARIFY
            ),
            forced_singleton=False,
        )
        rows.append(
            TurnDispositionCorpusRowV1(
                row_id=f"{trace_id}_start",
                trace_id=trace_id,
                tier=tier,
                turn_index=0,
                turn_kind="scored",
                is_state_query=False,
                legal_action_count=len(legal0.all_serialized_actions),
                coverage_complete=legal0.coverage is LegalSetCoverage.COMPLETE,
                forced_singleton=legal0.forced_action is not None,
                gold_pick=gold_pick,
                features=(margin, cheap_cost, exp_cost),
                target=target,
                split=split,
            )
        )

        # --- turn 1: p (forced singleton, non-query -> emit) ---
        p_source = f'root = TextContent(":t{n}.p")'
        _state1, legal1 = _state_and_legal_set(
            pack=pack, library=library, source=p_source, request_id=request_id, seed=2
        )
        rows.append(
            TurnDispositionCorpusRowV1(
                row_id=f"{trace_id}_p",
                trace_id=trace_id,
                tier=tier,
                turn_index=1,
                turn_kind="forced_emit",
                is_state_query=False,
                legal_action_count=len(legal1.all_serialized_actions),
                coverage_complete=legal1.coverage is LegalSetCoverage.COMPLETE,
                forced_singleton=legal1.forced_action is not None,
                gold_pick=None,
                features=(),
                target=None,
                split=split,
            )
        )

        # --- turn 2: mid (forced singleton, but is_state_query -> answer) ---
        mid_source = f'root = TextContent(":t{n}.mid")'
        _state2, legal2 = _state_and_legal_set(
            pack=pack, library=library, source=mid_source, request_id=request_id, seed=3
        )
        rows.append(
            TurnDispositionCorpusRowV1(
                row_id=f"{trace_id}_mid",
                trace_id=trace_id,
                tier=tier,
                turn_index=2,
                turn_kind="answer",
                is_state_query=True,
                legal_action_count=len(legal2.all_serialized_actions),
                coverage_complete=legal2.coverage is LegalSetCoverage.COMPLETE,
                forced_singleton=legal2.forced_action is not None,
                gold_pick=None,
                features=(),
                target=None,
                split=split,
            )
        )

        # --- turn 3: final (genuinely empty legal set -> out_of_scope) ---
        final_source = f'root = TextContent(":t{n}.final")'
        _state3, legal3 = _state_and_legal_set(
            pack=pack, library=library, source=final_source, request_id=request_id, seed=4
        )
        rows.append(
            TurnDispositionCorpusRowV1(
                row_id=f"{trace_id}_final",
                trace_id=trace_id,
                tier=tier,
                turn_index=3,
                turn_kind="out_of_scope",
                is_state_query=False,
                legal_action_count=len(legal3.all_serialized_actions),
                coverage_complete=legal3.coverage is LegalSetCoverage.COMPLETE,
                forced_singleton=legal3.forced_action is not None,
                gold_pick=None,
                features=(),
                target=None,
                split=split,
            )
        )

    rows_by_kind: dict[str, int] = {}
    rows_by_split: dict[str, int] = {}
    rows_by_tier: dict[int, int] = {}
    for row in rows:
        rows_by_kind[row.turn_kind] = rows_by_kind.get(row.turn_kind, 0) + 1
        rows_by_split[row.split] = rows_by_split.get(row.split, 0) + 1
        rows_by_tier[row.tier] = rows_by_tier.get(row.tier, 0) + 1

    quality_report = TurnDispositionCorpusQualityReportV1(
        total_traces=len(plan),
        total_rows=len(rows),
        rows_by_kind=rows_by_kind,
        rows_by_split=rows_by_split,
        rows_by_tier=rows_by_tier,
        out_of_scope_never_from_nonempty_set=all(
            row.legal_action_count == 0
            for row in rows
            if row.turn_kind == "out_of_scope"
        ),
    )
    return TurnDispositionCorpusV1(rows=tuple(rows), quality_report=quality_report)
