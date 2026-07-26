"""Fail-closed continuous program-latent objectives and traversal records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch
import torch.nn.functional as F

LOCKED_PROMOTION_METRICS = frozenset({"meaning_v2", "binder_reference_f1", "g0_g8"})


@dataclass(frozen=True)
class ProgramLatentObjective:
    total: torch.Tensor
    reconstruction: torch.Tensor
    contrast: torch.Tensor
    noise_consistency: torch.Tensor


def combined_program_latent_objective(
    reconstruction: torch.Tensor,
    positive_latents: torch.Tensor,
    negative_latents: torch.Tensor,
    *,
    contrast_margin: float,
    contrast_weight: float,
    noisy_latents: torch.Tensor | None = None,
    noise_weight: float = 0.0,
    training_metrics: tuple[str, ...] = (),
) -> ProgramLatentObjective:
    """Combine only representation losses; promotion metrics are never losses."""
    forbidden = sorted(set(training_metrics) & LOCKED_PROMOTION_METRICS)
    if forbidden:
        raise ValueError(f"locked promotion metrics cannot be training losses: {forbidden}")
    if contrast_margin < 0 or contrast_weight < 0 or noise_weight < 0:
        raise ValueError("objective weights and margin must be non-negative")
    distances = torch.linalg.vector_norm(positive_latents - negative_latents, dim=-1)
    contrast = F.relu(torch.as_tensor(contrast_margin, device=distances.device) - distances).mean()
    noise = torch.zeros((), device=positive_latents.device)
    if noisy_latents is not None:
        noise = F.mse_loss(noisy_latents, positive_latents)
    total = reconstruction + contrast_weight * contrast + noise_weight * noise
    return ProgramLatentObjective(total, reconstruction, contrast, noise)


def trace_interpolation(
    start: torch.Tensor,
    end: torch.Tensor,
    *,
    steps: int,
    decode_latent: Callable[[torch.Tensor], str] | None,
    verify_program: Callable[[str], Mapping[str, Any]] | None,
    factor_fingerprint: Callable[[str], Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Record every interpolation point and semantic boundary, never endpoints only."""
    if steps < 2:
        raise ValueError("steps must include both endpoints")
    if decode_latent is None or verify_program is None or factor_fingerprint is None:
        raise ValueError("decoder, verifier, and factor fingerprint callbacks are required")
    rows: list[dict[str, Any]] = []
    previous: Mapping[str, Any] | None = None
    for index in range(steps):
        alpha = index / (steps - 1)
        latent = torch.lerp(start, end, alpha)
        program = decode_latent(latent)
        verification = dict(verify_program(program))
        fingerprint = dict(factor_fingerprint(program))
        rows.append({
            "index": index,
            "alpha": alpha,
            "program": program,
            "verification": verification,
            "factor_fingerprint": fingerprint,
            "boundary_crossing": previous is not None and fingerprint != previous,
        })
        previous = fingerprint
    return rows
