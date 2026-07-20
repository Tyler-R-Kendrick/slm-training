"""SLM-150 (SPV2-02): global semantic critic candidate selector.

Wiring/fixture harness only. The selector composes a per-candidate local score
with the global semantic critic's energy (lower is better). It is a permutation-
only adapter: candidate set membership is never changed, and full abstention
passes through cleanly.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from slm_training.harnesses.experiments.candidate_selector import (
    SelectionCandidate,
    SelectionDecision,
)
from slm_training.models.global_semantic_critic import (
    GlobalSemanticCritic,
    SemanticEnergyOutput,
)

__all__ = ["GlobalSemanticCriticSelector"]


def _safe_local_score(candidate: SelectionCandidate) -> float:
    """Return a finite local score, preferring value_score then generator_score."""
    if candidate.value_score is not None and math.isfinite(candidate.value_score):
        return candidate.value_score
    if candidate.generator_score is not None and math.isfinite(candidate.generator_score):
        return candidate.generator_score
    return 0.0


class GlobalSemanticCriticSelector:
    """Select the candidate with the best ``local_score - lambda_global * energy``."""

    selector_id = "global_semantic_critic"

    def __init__(
        self,
        critic: GlobalSemanticCritic,
        *,
        lambda_global: float = 1.0,
        confidence_threshold: float = 0.5,
    ) -> None:
        self.critic = critic
        self.lambda_global = lambda_global
        self.confidence_threshold = confidence_threshold

    def select(
        self,
        *,
        prompt_context: Mapping[str, Any],
        structured_contract: Mapping[str, Any],
        candidates: Sequence[SelectionCandidate],
    ) -> SelectionDecision:
        """Pick the candidate with the highest combined global/local score."""
        if not candidates:
            return SelectionDecision(
                selected_candidate_id=None,
                abstained=True,
                fallback_policy="empty_candidate_set",
                predicted_success=None,
                utility_scores=(),
                selector_id=self.selector_id,
                threshold_id="",
                reason_code="empty_candidate_set",
            )

        semantic_plan = (
            prompt_context.get("semantic_plan", {})
            if isinstance(prompt_context, Mapping)
            else {}
        )
        context: dict[str, Any] = (
            dict(prompt_context) if isinstance(prompt_context, Mapping) else {}
        )
        if isinstance(structured_contract, Mapping):
            pack_id = structured_contract.get("pack_id")
            if pack_id is not None and "pack_id" not in context:
                context["pack_id"] = pack_id

        outputs: list[SemanticEnergyOutput] = []
        combined_scores: list[float] = []

        for candidate in candidates:
            candidate_features: dict[str, Any] = dict(candidate.available_features)
            candidate_features["candidate_id"] = candidate.candidate_id
            candidate_features["canonical_program"] = candidate.canonical_program
            candidate_features["ast_fingerprint"] = candidate.ast_fingerprint

            local = _safe_local_score(candidate)
            output = self.critic.score(
                context,
                semantic_plan,
                candidate_features,
                structured_contract,
            )
            outputs.append(output)
            if output.abstained:
                combined_scores.append(float("-inf"))
            else:
                combined_scores.append(local - self.lambda_global * output.energy)

        if all(output.abstained for output in outputs):
            return SelectionDecision(
                selected_candidate_id=None,
                abstained=True,
                fallback_policy="global_semantic_critic_abstain",
                predicted_success=None,
                utility_scores=tuple(combined_scores),
                selector_id=self.selector_id,
                threshold_id="",
                reason_code="all_abstained",
            )

        best_index = max(
            range(len(candidates)),
            key=lambda i: (combined_scores[i], candidates[i].candidate_id),
        )
        best_candidate = candidates[best_index]
        best_output = outputs[best_index]

        return SelectionDecision(
            selected_candidate_id=best_candidate.candidate_id,
            abstained=False,
            fallback_policy="global_semantic_critic",
            predicted_success=best_output.confidence,
            utility_scores=tuple(combined_scores),
            selector_id=self.selector_id,
            threshold_id="",
            reason_code="selected_global_semantic_critic",
        )
