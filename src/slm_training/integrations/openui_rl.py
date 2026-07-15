"""Shared honest OpenUI reward contract for external causal-LM RL engines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from slm_training.dsl.placeholders import extract_placeholders
from slm_training.harnesses.model_build.eval_runner import structural_similarity
from slm_training.harnesses.preference import grammar_score


@dataclass(frozen=True)
class OpenUIReward:
    parse: float
    placeholder_fidelity: float
    structural_similarity: float
    composite: float

    def to_dict(self) -> dict[str, float]:
        return {
            "parse": self.parse,
            "placeholder_fidelity": self.placeholder_fidelity,
            "structural_similarity": self.structural_similarity,
            "composite": self.composite,
        }


def score_openui(
    prediction: str,
    *,
    gold_openui: str,
    slot_inventory: Sequence[str],
) -> OpenUIReward:
    """Score only visible structure and the prompt's declared slot contract."""
    parse = grammar_score(prediction)
    predicted = set(extract_placeholders(prediction))
    expected = set(slot_inventory)
    if expected:
        fidelity = len(predicted & expected) / len(expected)
    else:
        fidelity = 1.0 if not predicted else 0.5
    structure = structural_similarity(prediction, gold_openui)
    composite = 0.0
    if parse > 0.0:
        composite = 0.45 * parse + 0.30 * fidelity + 0.25 * structure
    return OpenUIReward(
        parse=round(parse, 4),
        placeholder_fidelity=round(fidelity, 4),
        structural_similarity=round(structure, 4),
        composite=round(composite, 4),
    )
