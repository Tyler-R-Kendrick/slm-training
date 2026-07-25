"""Exact reference and bounded production samplers for legal-edit rates."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import torch

from slm_training.data.flow.bridge_corpus import (
    RequestEditContractV1,
    canonical_fingerprint,
    enumerate_live_candidates,
    parse_statements,
    verify_certificate,
)
from slm_training.dsl.parser import validate
from slm_training.flow.reference.adapter import StateRef
from slm_training.flow.reference.sampler import GillespieSampler
from slm_training.flow.reference.trajectory import FlowTrajectoryV1
from slm_training.flow.termination import (
    ABSTAIN,
    HOLD,
    FixedKPolicy,
    TerminationContext,
    TerminationPolicy,
)
from slm_training.harnesses.experiments.slm188_edit_algebra import (
    CanonicalEdit,
    apply_canonical_edit,
)
from slm_training.models.legal_edit_batch import LegalEditBatch
from slm_training.models.legal_edit_flow import LegalEditFlow


@dataclass(frozen=True)
class ProductionFlowTrace:
    final_program: str
    final_fingerprint: str
    stop_reason: str
    decisions: tuple[dict[str, Any], ...]
    verified_output: bool
    unknown_candidate_events: int
    elapsed_seconds: float


def sample_exact_reference(
    sampler: GillespieSampler,
    source: StateRef,
    rng: Any,
    *,
    terminal_check: Any,
) -> FlowTrajectoryV1:
    """Expose the frozen exact SLM-190 Gillespie implementation."""
    return sampler.sample(source, rng, terminal_check=terminal_check)


class ProductionLegalEditFlowSampler:
    """Bounded one-edit-at-a-time decode with live candidate refresh."""

    def __init__(self, model: LegalEditFlow) -> None:
        if not model.config.enabled:
            raise ValueError("production flow sampling requires explicit opt-in")
        self.model = model

    def sample(
        self,
        source: str,
        contract: RequestEditContractV1,
        *,
        termination: TerminationPolicy | None = None,
        max_steps: int = 4,
        max_wall_seconds: float = 30.0,
        seed: int = 0,
        final_verifier: Callable[[str], bool] | None = None,
    ) -> ProductionFlowTrace:
        termination = termination or FixedKPolicy(k=max_steps, max_steps=max_steps)
        generator = torch.Generator().manual_seed(seed)
        current = source.strip()
        started = time.monotonic()
        trace: list[dict[str, Any]] = []
        unknown_events = 0
        stop_reason = ABSTAIN
        device = next(self.model.parameters()).device
        for step in range(max_steps + 1):
            elapsed = time.monotonic() - started
            try:
                candidate_set = enumerate_live_candidates(current, contract)
            except Exception:  # noqa: BLE001
                stop_reason = "candidate_enumeration_unknown"
                break
            statements = parse_statements(current) or ()
            batch = LegalEditBatch.pack_inference(
                candidate_set,
                statement_count=len(statements),
                step_index=step,
                device=device,
            )
            unknown_events += int(batch.unknown_mask.sum())
            prediction = None
            hazard = 0.0
            if batch.candidate_ids:
                with torch.no_grad():
                    prediction = self.model(
                        batch,
                        schedule_progress=torch.tensor(
                            [step / max(1, max_steps)],
                            dtype=torch.float32,
                            device=device,
                        ),
                    )
                hazard = float(prediction.row_hazards[0])
                if (
                    not bool(torch.isfinite(prediction.edge_rates).all())
                    or not bool(torch.isfinite(prediction.row_hazards).all())
                    or hazard <= 0.0
                ):
                    stop_reason = "rate_prediction_unknown"
                    break
            decision = termination.decide(
                TerminationContext(
                    state_fingerprint=canonical_fingerprint(current),
                    step_index=step,
                    edit_count=step,
                    wall_time=elapsed,
                    total_hazard=hazard,
                    candidates=batch.candidate_ids,
                )
            )
            if decision.action != HOLD:
                stop_reason = decision.reason
                break
            if elapsed > max_wall_seconds or not batch.candidate_ids:
                stop_reason = "wall_budget" if elapsed > max_wall_seconds else ABSTAIN
                break
            assert prediction is not None
            probabilities = prediction.edge_rates / prediction.edge_rates.sum()
            selected = int(torch.multinomial(probabilities, 1, generator=generator))
            selected_id = batch.candidate_ids[selected]
            candidate = next(
                item
                for item in candidate_set.candidates
                if item.candidate_id == selected_id
            )
            try:
                verify_certificate(candidate.transition_certificate)
                successor = apply_canonical_edit(
                    current, CanonicalEdit.from_dict(candidate.edit)
                )
            except Exception:  # noqa: BLE001
                stop_reason = "invalid_replay"
                break
            if (
                successor is None
                or canonical_fingerprint(successor) != candidate.successor_fingerprint
            ):
                stop_reason = "invalid_replay"
                break
            trace.append(
                {
                    "step": step,
                    "candidate_set_digest": candidate_set.candidate_set_digest,
                    "candidate_ids": list(batch.candidate_ids),
                    "selected_candidate_id": candidate.candidate_id,
                    "selected_rate": float(prediction.edge_rates[selected]),
                    "total_hazard": hazard,
                    "successor_fingerprint": candidate.successor_fingerprint,
                }
            )
            current = successor
        parser_valid = False
        try:
            validate(current)
            parser_valid = True
        except Exception:  # noqa: BLE001
            stop_reason = "invalid_output"
        verified = parser_valid
        if parser_valid and final_verifier is not None:
            try:
                verified = bool(final_verifier(current))
            except Exception:  # noqa: BLE001
                verified = False
                stop_reason = "final_verification_unknown"
        if not verified and stop_reason not in {
            "candidate_enumeration_unknown",
            "invalid_output",
            "final_verification_unknown",
        }:
            stop_reason = "final_verification_unknown"
        try:
            final_fingerprint = canonical_fingerprint(current)
        except Exception:  # noqa: BLE001
            final_fingerprint = ""
        return ProductionFlowTrace(
            final_program=current,
            final_fingerprint=final_fingerprint,
            stop_reason=stop_reason,
            decisions=tuple(trace),
            verified_output=verified,
            unknown_candidate_events=unknown_events,
            elapsed_seconds=time.monotonic() - started,
        )
