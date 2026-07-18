"""Cost-to-go energy ranker for the verified solver (VSS3-02 / SLM-70).

The ranker has no authority over candidate membership, support, or final
verification. It only orders the exact live candidates supplied by the solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from slm_training.dsl.solver.controller import CandidateRanker
from slm_training.dsl.solver.state import DomainValue, FiniteDomainState, HoleId

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class CandidateEnergyInput:
    """Features for one scoring call."""

    state_fingerprint: str
    state_features: torch.Tensor
    hole_features: torch.Tensor
    candidate_ids: tuple[DomainValue, ...]
    candidate_features: torch.Tensor


@dataclass(frozen=True)
class CandidateEnergyOutput:
    """Energy scores for the supplied candidates."""

    energies: torch.Tensor  # shape [num_live_candidates]
    candidate_ids: tuple[DomainValue, ...]
    scorer_id: str


@runtime_checkable
class CandidateEnergyScorer(Protocol):
    """Callable that scores live candidates without altering membership."""

    def __call__(
        self,
        state: FiniteDomainState,
        hole_id: HoleId,
        values: tuple[DomainValue, ...],
    ) -> CandidateEnergyOutput: ...


class EnergyCandidateRanker(CandidateRanker):
    """Permutation-only ranker backed by an energy scorer.

    Falls back to the canonical value order when the scorer returns an invalid
    result (wrong length, non-finite values, or membership change).
    """

    def __init__(
        self,
        scorer: CandidateEnergyScorer,
        *,
        ranker_id: str | None = None,
        fallback: str = "deterministic",
    ) -> None:
        self._scorer = scorer
        self._ranker_id = ranker_id or f"energy-{getattr(scorer, 'scorer_id', id(scorer))}"
        self._fallback = fallback
        self._fallback_count = 0

    @property
    def ranker_id(self) -> str:
        return self._ranker_id

    @property
    def fallback_count(self) -> int:
        return self._fallback_count

    def rank(
        self,
        state: FiniteDomainState,
        hole_id: HoleId,
        values: tuple[DomainValue, ...],
    ) -> tuple[DomainValue, ...]:
        import torch

        if not values:
            return tuple(values)
        try:
            output = self._scorer(state, hole_id, values)
        except Exception:  # noqa: BLE001
            self._fallback_count += 1
            return tuple(values)

        if not isinstance(output, CandidateEnergyOutput):
            self._fallback_count += 1
            return tuple(values)

        if len(output.candidate_ids) != len(values) or len(output.energies) != len(
            values
        ):
            self._fallback_count += 1
            return tuple(values)

        if set(output.candidate_ids) != set(values):
            self._fallback_count += 1
            return tuple(values)

        if not torch.isfinite(output.energies).all():
            self._fallback_count += 1
            return tuple(values)

        # Lower energy ranks earlier.
        order = torch.argsort(output.energies, stable=True)
        ranked = tuple(output.candidate_ids[int(i)] for i in order.tolist())
        if set(ranked) != set(values) or len(ranked) != len(values):
            self._fallback_count += 1
            return tuple(values)
        return ranked


def make_stub_energy_scorer(
    scorer_id: str = "stub-v1",
) -> CandidateEnergyScorer:
    """Return a deterministic scorer that assigns lower energy to shorter payloads.

    Useful for fixture tests: it never touches a model and always returns finite
    energies for the exact supplied values.
    """

    def scorer(
        state: FiniteDomainState,
        hole_id: HoleId,
        values: tuple[DomainValue, ...],
    ) -> CandidateEnergyOutput:
        import torch

        energies = torch.tensor(
            [float(len(str(value.payload_json))) for value in values],
            dtype=torch.float32,
        )
        return CandidateEnergyOutput(
            energies=energies,
            candidate_ids=values,
            scorer_id=scorer_id,
        )

    return scorer


__all__ = [
    "CandidateEnergyInput",
    "CandidateEnergyOutput",
    "CandidateEnergyScorer",
    "EnergyCandidateRanker",
    "make_stub_energy_scorer",
]
