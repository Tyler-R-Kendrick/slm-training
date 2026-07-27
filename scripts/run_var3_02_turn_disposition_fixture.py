#!/usr/bin/env python3
"""VAR3-02 (SLM-430) fixture measurement: disposition-on vs. disposition-off.

Builds one deterministic, seeded synthetic corpus of turns (out_of_scope /
state-query / forced-singleton / ambiguous-margin / confident-margin cases)
and scores two matched-budget arms over the identical corpus and identical
(imperfect) candidate scores:

* ``disposition_off`` -- the pre-VAR3-02 baseline: always attempts to EMIT
  the top-scored candidate, with no concept of ``out_of_scope``, ``answer``,
  or a clarify margin. A forced EMIT against an empty legal set or a
  state-query turn is definitionally wrong (there is no legal op to emit).
* ``disposition_on`` -- ``dsl.operators.turn_disposition.derive_turn_disposition``
  with the registered levers
  (``levers.TURN_DISPOSITION_CLARIFY_MARGIN_THRESHOLD`` /
  ``levers.TURN_DISPOSITION_CLARIFY_TOP_K``).

Both arms are scored with ``evals.cap2_operator.score_disposition_predictions``
(``wrong_op_rate``, ``abstention_rate``, and the penalty-ratio composite).

This is fixture/synthetic-scale capability evidence, not a ship claim: the
synthetic score distribution is hand-authored to exercise every disposition
branch, not sampled from a real trained policy or a real corpus. Honesty
label: ``fixture_or_scratch``.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from slm_training.dsl.operators.contracts import _fingerprint
from slm_training.dsl.operators.legal_set import LegalSetCoverage, OperatorLegalSetV1
from slm_training.dsl.operators.turn_disposition import (
    TurnDispositionKind,
    derive_turn_disposition,
)
from slm_training.evals.cap2_operator import score_disposition_predictions
from slm_training.levers import (
    TURN_DISPOSITION_CLARIFY_MARGIN_THRESHOLD,
    TURN_DISPOSITION_CLARIFY_TOP_K,
    TURN_DISPOSITION_WRONG_EMIT_PENALTY_RATIO,
)
from slm_training.versioning import build_version_stamp

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs/design/var3-02-turn-disposition-20260726.json"
SEED = 430
CASE_COUNT = 200


def _digest(label: str) -> str:
    return _fingerprint({"schema": "var3_02_fixture/v1", "label": label})


def _legal_set(
    *, ordinary: tuple[str, ...], coverage: LegalSetCoverage = LegalSetCoverage.COMPLETE
) -> OperatorLegalSetV1:
    return OperatorLegalSetV1(
        state_fingerprint=_digest("state"),
        reference_table_fingerprint=_digest("refs"),
        registry_fingerprint=_digest("registry"),
        entries=(),
        ordinary_nonoperator_actions=ordinary,
        coverage=coverage,
        max_combinations_per_operator=10,
        max_rejection_samples_per_operator=0,
    )


class _SyntheticCase:
    __slots__ = ("legal_set", "is_state_query", "gold_action", "scores")

    def __init__(self, legal_set, is_state_query, gold_action, scores):
        self.legal_set = legal_set
        self.is_state_query = is_state_query
        self.gold_action = gold_action
        self.scores = scores


def build_synthetic_corpus(*, n: int, seed: int) -> list[_SyntheticCase]:
    """Deterministic, seeded synthetic corpus covering every disposition branch.

    Category mix (fractions of ``n``, in this fixed order so the corpus is
    reproducible bit-for-bit given ``seed``):

    - 10% ``out_of_scope``: empty legal set.
    - 15% state-query (``answer``): non-empty legal set, but the turn asks
      about state rather than proposing a mutation.
    - 15% forced singleton: exactly one legal candidate (I1 bypass).
    - 30% ambiguous margin: 3-5 candidates, top-1/runner-up margin drawn
      below the registered clarify threshold most of the time.
    - 30% confident margin: 3-5 candidates, margin drawn above threshold,
      but the top-scored candidate is only correct ~85% of the time
      (an imperfect scorer, not an oracle).
    """
    rng = random.Random(seed)
    cases: list[_SyntheticCase] = []
    n_out_of_scope = int(n * 0.10)
    n_query = int(n * 0.15)
    n_forced = int(n * 0.15)
    n_ambiguous = int(n * 0.30)
    n_confident = n - n_out_of_scope - n_query - n_forced - n_ambiguous

    for i in range(n_out_of_scope):
        cases.append(_SyntheticCase(_legal_set(ordinary=()), False, None, {}))

    for i in range(n_query):
        actions = (f"OPERATOR query_{i}_a", f"OPERATOR query_{i}_b")
        cases.append(
            _SyntheticCase(
                _legal_set(ordinary=actions), True, None, {a: 0.5 for a in actions}
            )
        )

    for i in range(n_forced):
        action = f"OPERATOR forced_{i}"
        cases.append(
            _SyntheticCase(_legal_set(ordinary=(action,)), False, action, {action: 1.0})
        )

    for i in range(n_ambiguous):
        k = rng.choice((3, 4, 5))
        actions = tuple(f"OPERATOR ambiguous_{i}_{j}" for j in range(k))
        gold = rng.choice(actions)
        top_score = rng.uniform(0.5, 0.7)
        # Margin drawn below the registered threshold ~90% of the time.
        margin = (
            rng.uniform(0.0, TURN_DISPOSITION_CLARIFY_MARGIN_THRESHOLD * 0.6)
            if rng.random() < 0.9
            else rng.uniform(
                TURN_DISPOSITION_CLARIFY_MARGIN_THRESHOLD, TURN_DISPOSITION_CLARIFY_MARGIN_THRESHOLD + 0.2
            )
        )
        scores = {a: rng.uniform(0.0, top_score - margin - 0.01) for a in actions}
        best = rng.choice(actions)
        scores[best] = top_score
        second = rng.choice([a for a in actions if a != best])
        scores[second] = max(0.0, top_score - margin)
        cases.append(_SyntheticCase(_legal_set(ordinary=actions), False, gold, scores))

    for i in range(n_confident):
        k = rng.choice((3, 4, 5))
        actions = tuple(f"OPERATOR confident_{i}_{j}" for j in range(k))
        gold = rng.choice(actions)
        top_score = rng.uniform(0.8, 0.99)
        margin = rng.uniform(
            TURN_DISPOSITION_CLARIFY_MARGIN_THRESHOLD, TURN_DISPOSITION_CLARIFY_MARGIN_THRESHOLD + 0.3
        )
        scores = {a: rng.uniform(0.0, max(0.0, top_score - margin - 0.01)) for a in actions}
        # The scorer is correct ~85% of the time -- an imperfect model, not an oracle.
        top_choice = gold if rng.random() < 0.85 else rng.choice([a for a in actions if a != gold])
        scores[top_choice] = top_score
        cases.append(_SyntheticCase(_legal_set(ordinary=actions), False, gold, scores))

    rng.shuffle(cases)
    return cases


def _disposition_off_row(case: _SyntheticCase) -> dict[str, Any]:
    """Pre-VAR3-02 baseline: always attempts EMIT of the top-scored candidate."""
    actions = case.legal_set.all_serialized_actions
    if not actions or case.is_state_query:
        # No legal candidate to emit (out_of_scope) or emitting against a
        # read-only query turn -- definitionally wrong.
        return {"predicted_disposition": "emit", "emit_correct": False}
    top = max(actions, key=lambda a: case.scores.get(a, 0.0))
    return {"predicted_disposition": "emit", "emit_correct": top == case.gold_action}


def _disposition_on_row(case: _SyntheticCase) -> dict[str, Any]:
    result = derive_turn_disposition(
        legal_set=case.legal_set,
        is_state_query=case.is_state_query,
        clarify_margin_threshold=TURN_DISPOSITION_CLARIFY_MARGIN_THRESHOLD,
        clarify_top_k=TURN_DISPOSITION_CLARIFY_TOP_K,
        score_provider=lambda: case.scores,
    )
    if result.kind is TurnDispositionKind.EMIT:
        return {
            "predicted_disposition": "emit",
            "emit_correct": result.chosen_action == case.gold_action,
        }
    if result.kind is TurnDispositionKind.CLARIFY:
        return {"predicted_disposition": "clarify"}
    if result.kind is TurnDispositionKind.ANSWER:
        return {"predicted_disposition": "answer"}
    return {"predicted_disposition": "out_of_scope"}


def run_fixture(*, n: int = CASE_COUNT, seed: int = SEED) -> dict[str, Any]:
    corpus = build_synthetic_corpus(n=n, seed=seed)
    off_rows = [_disposition_off_row(case) for case in corpus]
    on_rows = [_disposition_on_row(case) for case in corpus]
    off_score = score_disposition_predictions(off_rows)
    on_score = score_disposition_predictions(on_rows)
    return {
        "schema": "var3_02_turn_disposition_fixture/v1",
        "honesty": "fixture_or_scratch",
        "claim_class": "capability",
        "case_count": n,
        "seed": seed,
        "clarify_margin_threshold": TURN_DISPOSITION_CLARIFY_MARGIN_THRESHOLD,
        "clarify_top_k": TURN_DISPOSITION_CLARIFY_TOP_K,
        "wrong_emit_penalty_ratio": TURN_DISPOSITION_WRONG_EMIT_PENALTY_RATIO,
        "arms": {
            "disposition_off": off_score,
            "disposition_on": on_score,
        },
        "version_stamp": build_version_stamp(
            "config.levers",
            "dsl.operators.turn_disposition",
            "dsl.operators.conversation",
            "evals.cap2_operator",
            "model.turn_disposition_head",
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--case-count", type=int, default=CASE_COUNT)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    report = run_fixture(n=args.case_count, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
