"""SLM-433 (VAR3-03): train TurnDispositionHead on a real corpus, measure
held-out benefit against two matched non-learned baselines.

Three matched arms are scored on the identical held-out (trace-leakage-safe)
split via ``evals.cap2_operator.score_disposition_predictions`` (reused, not
reimplemented):

- ``disposition_off`` -- the pre-VAR3-02 baseline: always attempts to
  ``emit`` the top real-cost-ranked candidate, with no ``out_of_scope`` /
  ``answer`` / clarify-margin concept at all.
- ``derived_only`` -- correctly resolves ``out_of_scope`` / ``answer`` /
  forced-singleton via the same free, non-learned rules
  ``dsl.operators.turn_disposition.derive_turn_disposition`` uses, but never
  clarifies: every remaining ("scored") turn still just emits the top
  real-cost-ranked candidate.
- ``disposition_trained`` -- identical free-rule resolution for
  ``out_of_scope`` / ``answer`` / forced-singleton, but a
  ``TurnDispositionHead``-shaped classifier (trained via
  ``models.turn_disposition_head.turn_disposition_losses``, never a
  reimplemented loss) decides clarify-vs-emit on the remaining turns.

The classifier is architecturally never consulted for ``out_of_scope`` /
``answer`` / forced-singleton turns (I1 precedence is unchanged); its
``answer`` logit is masked to -inf at the one decision point it is ever
invoked (post out_of_scope/answer/forced-singleton), since deployment
context there has already ruled ``answer`` out.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from slm_training.data.flow.turn_disposition_corpus import (
    TIER_GOLD_CHEAP_PROBABILITY,
    TIER_MARGINS,
    TurnDispositionCorpusRowV1,
    TurnDispositionCorpusV1,
    build_turn_disposition_corpus,
)
from slm_training.evals.cap2_operator import score_disposition_predictions
from slm_training.models.turn_disposition_head import (
    TurnDispositionLabel,
    turn_disposition_losses,
)
from slm_training.versioning import build_version_stamp

EXPERIMENT_ID = "slm433-turn-disposition-corpus"
ARMS = ("disposition_off", "derived_only", "disposition_trained")

_LABEL_ORDER = (
    TurnDispositionLabel.ANSWER,
    TurnDispositionLabel.CLARIFY,
    TurnDispositionLabel.EMIT,
)
_ANSWER_INDEX = _LABEL_ORDER.index(TurnDispositionLabel.ANSWER)
_CLARIFY_INDEX = _LABEL_ORDER.index(TurnDispositionLabel.CLARIFY)
_EMIT_INDEX = _LABEL_ORDER.index(TurnDispositionLabel.EMIT)

PREREGISTERED = {
    "hypothesis": (
        "A TurnDispositionHead trained on real, corpus-derived clarify-margin "
        "examples reduces composite_penalized_error_rate on a held-out split "
        "relative to derived_only (free rules, clarify never fires)."
    ),
    "stop_rule": (
        "If disposition_trained shows no held-out improvement over "
        "derived_only, that is a legitimate negative result; do not add "
        "further trained-head complexity to this surface."
    ),
    "leakage_safety": "split is trace-level; all four rows of one trace share one split",
}


def _classifier_module():
    import torch
    import torch.nn as nn

    class _DispositionClassifier(nn.Module):
        def __init__(self, n_features: int = 3, hidden: int = 16) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_features, hidden),
                nn.Tanh(),
                nn.Linear(hidden, len(_LABEL_ORDER)),
            )

        def forward(self, features: "torch.Tensor") -> "torch.Tensor":
            return self.net(features)

    return _DispositionClassifier()


def train_disposition_head(
    scored_train_rows: tuple[TurnDispositionCorpusRowV1, ...],
    *,
    steps: int = 200,
    lr: float = 0.05,
    seed: int = 433,
):
    """Train a small classifier via ``turn_disposition_losses`` -- reused,
    not reimplemented. Returns the trained ``nn.Module``."""
    import torch

    torch.manual_seed(seed)
    model = _classifier_module()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rows = list(scored_train_rows)
    rng = random.Random(seed)
    model.train()
    total_loss = 0.0
    for _ in range(steps):
        rng.shuffle(rows)
        for row in rows:
            assert row.target is not None
            features = torch.tensor([row.features], dtype=torch.float32)
            logits = model(features).squeeze(0)
            loss, _ = turn_disposition_losses(logits, row.target, mask_forced=True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss = float(loss.detach())
    model.eval()
    return model, total_loss


def _predict_clarify(model, row: TurnDispositionCorpusRowV1) -> bool:
    """Query the trained head; ``answer`` is masked out -- this decision
    point is only ever reached after out_of_scope/answer/forced-singleton
    have already been ruled out by the free derivations (I1 unchanged)."""
    import torch

    with torch.no_grad():
        features = torch.tensor([row.features], dtype=torch.float32)
        logits = model(features).squeeze(0).clone()
        logits[_ANSWER_INDEX] = float("-inf")
        predicted_index = int(torch.argmax(logits).item())
    return predicted_index == _CLARIFY_INDEX


def _naive_emit_correct(row: TurnDispositionCorpusRowV1) -> bool:
    """The blind/derived_only scored-row policy: always emit the cheaper
    (top-ranked-by-real-cost) candidate."""
    return row.gold_pick == "cheap"


def _score_arm(
    corpus: TurnDispositionCorpusV1, *, arm: str, model=None
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for row in corpus.by_split("heldout"):
        if row.turn_kind == "out_of_scope":
            if arm == "disposition_off":
                rows.append({"predicted_disposition": "emit", "emit_correct": False})
            else:
                rows.append({"predicted_disposition": "out_of_scope", "emit_correct": None})
        elif row.turn_kind == "answer":
            if arm == "disposition_off":
                rows.append({"predicted_disposition": "emit", "emit_correct": False})
            else:
                rows.append({"predicted_disposition": "answer", "emit_correct": None})
        elif row.turn_kind == "forced_emit":
            rows.append({"predicted_disposition": "emit", "emit_correct": True})
        else:
            assert row.turn_kind == "scored"
            if arm == "disposition_trained":
                assert model is not None
                if _predict_clarify(model, row):
                    rows.append({"predicted_disposition": "clarify", "emit_correct": None})
                    continue
            rows.append(
                {
                    "predicted_disposition": "emit",
                    "emit_correct": _naive_emit_correct(row),
                }
            )
    return score_disposition_predictions(rows)


@dataclass(frozen=True)
class Slm433Report:
    payload: dict[str, Any]


def run_experiment(
    *,
    n_per_tier: int = 40,
    train_fraction: float = 0.7,
    corpus_seed: int = 433,
    train_steps: int = 200,
    train_lr: float = 0.05,
    train_seed: int = 433,
) -> dict[str, Any]:
    corpus = build_turn_disposition_corpus(
        n_per_tier=n_per_tier, train_fraction=train_fraction, seed=corpus_seed
    )
    scored_train = corpus.scored("train")
    scored_heldout = corpus.scored("heldout")
    if not scored_train or not scored_heldout:
        raise ValueError("corpus produced no scored rows for train or heldout split")

    model, final_train_loss = train_disposition_head(
        scored_train, steps=train_steps, lr=train_lr, seed=train_seed
    )

    arm_scores = {
        "disposition_off": _score_arm(corpus, arm="disposition_off"),
        "derived_only": _score_arm(corpus, arm="derived_only"),
        "disposition_trained": _score_arm(corpus, arm="disposition_trained", model=model),
    }

    trained_composite = arm_scores["disposition_trained"]["composite_penalized_error_rate"]
    derived_composite = arm_scores["derived_only"]["composite_penalized_error_rate"]
    improved = trained_composite < derived_composite
    disposition = "trained_head_improves_on_derived_only" if improved else "no_held_out_improvement"

    # Trace-level leakage-safety proof: no trace_id appears in both splits.
    train_traces = {row.trace_id for row in corpus.by_split("train")}
    heldout_traces = {row.trace_id for row in corpus.by_split("heldout")}
    leakage_safe = train_traces.isdisjoint(heldout_traces)

    payload: dict[str, Any] = {
        "experiment": EXPERIMENT_ID,
        "issue": "SLM-433",
        "preregistered": PREREGISTERED,
        "arms": list(ARMS),
        "corpus_config": {
            "n_per_tier": n_per_tier,
            "train_fraction": train_fraction,
            "corpus_seed": corpus_seed,
            "tier_margins": list(TIER_MARGINS),
            "tier_gold_cheap_probability": list(TIER_GOLD_CHEAP_PROBABILITY),
        },
        "quality_report": corpus.quality_report.to_dict(),
        "leakage_safe_split": leakage_safe,
        "train_scored_count": len(scored_train),
        "heldout_scored_count": len(scored_heldout),
        "final_train_loss": final_train_loss,
        "arm_scores": arm_scores,
        "disposition": disposition,
        "honesty": (
            "fixture_or_scratch, capability class -- not a ship claim. The "
            "corpus is a real, freshly-enumerated symbolic fixture (5 real "
            "declared-cost tiers x n_per_tier traces), not natural-language "
            "data; the classifier is a tiny 3-feature MLP. Held-out split is "
            "trace-level leakage-safe. No promotion, ship-gate, or "
            "production-readiness claim is made regardless of outcome."
        ),
    }
    payload["version_stamp"] = build_version_stamp(
        "harness.experiments.slm433_turn_disposition_corpus",
        "model.turn_disposition_head",
        "dsl.operators.turn_disposition",
        "data.flow.turn_disposition_corpus",
    )
    return payload
