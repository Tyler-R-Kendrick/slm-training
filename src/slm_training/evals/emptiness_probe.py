"""A1 emptiness probe: is valid-but-empty a constraint-distortion artifact?

MODEL_CARD E224-E236 report syntax parse = 1.0 with meaningful parse ~= 0: the
constrained decoder reliably emits grammatically valid but trivial/empty
layouts. Grammar-Aligned Decoding / ASAp (Park et al., NeurIPS 2024) shows that
picking the model-highest *grammar-valid* completion distorts the model's
distribution; combined with a length prior it can make the shortest valid
program (the empty document) the argmax. This module tests that hypothesis
directly, with no training.

For a held-out record we score the model's fully-masked (MaskGIT first-step,
mean-field) reconstruction NLL of two grammar-valid programs conditioned on the
same context:

* ``y_pop`` — the gold populated program (``ExampleRecord.openui``);
* ``y_empty`` — a deterministic *minimal valid* program for the active DSL.

We report the comparison two ways:

* **total** sequence NLL — what a score-ranking constrained decoder actually
  compares; shorter programs are cheaper here, so this exposes the length bias;
* **per-token** NLL — length-controlled; if the populated program is cheaper
  per token the model *does* prefer real content locally.

Verdict decomposition (the point of the probe):

* empty wins on **total** but populated wins (or ties) on **per-token**
  => length-bias / constraint distortion. The emptiness is a *decode-time*
  artifact; fix it in Track A (A2 distribution-aware decode, A4 min-content
  contracts, E2 semantic-density gates), not with more training.
* empty wins on **per-token** too => the model genuinely does not model
  content; representation/training work (Tracks B/C/D) is required.

This is a diagnostic, never a ship metric. Fixture/scratch checkpoints produce
wiring evidence only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from slm_training.dsl.schema import ExampleRecord
from slm_training.evals.denoising_nll import (
    LOSS_SUITE_VERSION,
    _context_text,
    _target_ids,
    default_eligible_positions,
)
from slm_training.evals.score_policy import (
    CandidatePath,
    ScorePolicy,
    compare_policies,
)

# Deterministic minimal-document candidates, tried in order; the first that
# validates for the active DSL is the empty baseline. Kept tiny and structural
# so the comparison isolates "content vs no content", not surface style.
MINIMAL_CANDIDATES: tuple[str, ...] = (
    "root = Stack([])",
    'root = Stack([], "column")',
    "root = Container([])",
    "root = Fragment([])",
    "root = Box([])",
)


@dataclass(frozen=True)
class EmptinessProbeConfig:
    suite_version: str = LOSS_SUITE_VERSION
    batch_size: int = 8
    # Per-token margin below this (nats) counts as "populated preferred locally".
    tie_epsilon: float = 0.02
    # Optional eval-only score policies to compare on the same two candidates.
    score_policies: tuple[ScorePolicy, ...] = ()


def minimal_valid_program(dsl: str | None = None) -> str | None:
    """First :data:`MINIMAL_CANDIDATES` entry that validates for ``dsl``.

    Returns ``None`` when no minimal template is valid for the active grammar
    (the probe then honestly skips every record rather than inventing a
    baseline).
    """
    from slm_training.dsl.parser import validate

    for candidate in MINIMAL_CANDIDATES:
        try:
            validate(candidate, dsl=dsl)
        except Exception:  # noqa: BLE001 - any parse/validate failure disqualifies
            continue
        return candidate
    return None


def _encode_text(model: Any, text: str, cache_key: str) -> list[int]:
    """Encode an arbitrary OpenUI program the same way targets are encoded."""
    from slm_training.models.twotower import _truncate_with_eos

    ids = model._encode_openui(text, placeholders=[], cache_key=cache_key)
    return _truncate_with_eos(
        ids, int(model.config.max_target_len), model.tokenizer.eos_id
    )


@torch.no_grad()
def _sequence_log_probs(
    model: Any, context_text: str, ids: list[int]
) -> tuple[list[float], list[int]]:
    """Return per-position gold-token log-probabilities and their positions."""
    tokenizer = model.tokenizer
    positions = default_eligible_positions(model, ids)
    if not positions:
        return [], []
    ctx, ctx_pad = model._encode_context([context_text], cache_keys=None)
    row = torch.tensor([ids], dtype=torch.long, device=model.device_name)
    noisy = row.clone()
    for pos in positions:
        noisy[0, pos] = tokenizer.mask_id
    logits = model.denoiser(noisy, ctx, pad_id=tokenizer.pad_id, ctx_pad_mask=ctx_pad)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    token_lp = log_probs.gather(-1, row.unsqueeze(-1)).squeeze(-1)
    lps = [float(token_lp[0, pos].item()) for pos in positions]
    return lps, positions


@torch.no_grad()
def _sequence_nll(
    model: Any, context_text: str, ids: list[int]
) -> tuple[float, int]:
    """Mean-field (all-mask) reconstruction NLL of ``ids`` given ``context``.

    Masks every predictable position at once and scores the gold tokens under a
    single denoiser forward — the diffusion analogue of a from-scratch
    full-program score, and the quantity a score-ranking constrained decoder
    compares across candidate completions.
    """
    lps, positions = _sequence_log_probs(model, context_text, ids)
    if not positions:
        return 0.0, 0
    return float(-sum(lps)), len(positions)


@torch.no_grad()
def evaluate_emptiness(
    model: Any,
    records: list[ExampleRecord],
    *,
    config: EmptinessProbeConfig | None = None,
    dsl: str | None = None,
) -> dict[str, Any]:
    """Score populated vs minimal-valid programs across ``records``.

    Aggregates the fraction of records where the empty program is the cheaper
    (preferred) completion on total NLL and on per-token NLL, plus mean margins,
    and returns a verdict per the module docstring's decomposition.
    """
    cfg = config or EmptinessProbeConfig()
    was_training = bool(getattr(model, "training", False))
    model.eval()

    empty_text = minimal_valid_program(dsl)
    if empty_text is None:
        if was_training:
            model.train()
        return {
            "suite_version": cfg.suite_version,
            "n_records": 0,
            "n_skipped": len(records),
            "empty_program": None,
            "reason": "no minimal valid program for active DSL",
            "per_record": [],
        }

    per_record: list[dict[str, Any]] = []
    empty_pref_total = 0
    empty_pref_per_token = 0
    margin_total_sum = 0.0
    margin_per_token_sum = 0.0
    scored = 0
    skipped = 0
    policy_empty_preferred: dict[str, int] = {p.name: 0 for p in cfg.score_policies}

    for record in records:
        pop_ids = _target_ids(model, record)
        empty_ids = _encode_text(model, empty_text, cache_key=f"empty::{record.id}")
        context = _context_text(model, record)
        pop_total, pop_n = _sequence_nll(model, context, pop_ids)
        empty_total, empty_n = _sequence_nll(model, context, empty_ids)
        pop_lps, pop_positions = _sequence_log_probs(model, context, pop_ids)
        empty_lps, empty_positions = _sequence_log_probs(model, context, empty_ids)
        if pop_n == 0 or empty_n == 0:
            skipped += 1
            continue
        scored += 1
        pop_per_token = pop_total / pop_n
        empty_per_token = empty_total / empty_n
        # margin > 0 => empty is cheaper (preferred) => the emptiness pull.
        margin_total = pop_total - empty_total
        margin_per_token = pop_per_token - empty_per_token
        empty_wins_total = margin_total > 0
        empty_wins_per_token = margin_per_token > cfg.tie_epsilon
        empty_pref_total += int(empty_wins_total)
        empty_pref_per_token += int(empty_wins_per_token)
        margin_total_sum += margin_total
        margin_per_token_sum += margin_per_token

        rec: dict[str, Any] = {
            "id": record.id,
            "split": record.split,
            "pop_tokens": pop_n,
            "empty_tokens": empty_n,
            "pop_nll_total": pop_total,
            "empty_nll_total": empty_total,
            "pop_nll_per_token": pop_per_token,
            "empty_nll_per_token": empty_per_token,
            "margin_total": margin_total,
            "margin_per_token": margin_per_token,
            "empty_preferred_total": empty_wins_total,
            "empty_preferred_per_token": empty_wins_per_token,
        }

        if cfg.score_policies:
            pop_path = CandidatePath(
                candidate_id="populated",
                token_ids=tuple(pop_ids[pos] for pos in pop_positions),
                log_probs=tuple(pop_lps),
            )
            empty_path = CandidatePath(
                candidate_id="empty",
                token_ids=tuple(empty_ids[pos] for pos in empty_positions),
                log_probs=tuple(empty_lps),
            )
            policy_comparison = compare_policies(
                [pop_path, empty_path], cfg.score_policies
            )
            rec["policy_scores"] = policy_comparison["scores"]
            rec["policy_rankings"] = policy_comparison["rankings"]
            for policy_name, ranking in policy_comparison["rankings"].items():
                if ranking and ranking[0] == "empty":
                    policy_empty_preferred[policy_name] += 1

        per_record.append(rec)

    if was_training:
        model.train()

    frac_total = empty_pref_total / scored if scored else None
    frac_per_token = empty_pref_per_token / scored if scored else None
    verdict = _verdict(frac_total, frac_per_token)
    result: dict[str, Any] = {
        "suite_version": cfg.suite_version,
        "n_records": scored,
        "n_skipped": skipped,
        "empty_program": empty_text,
        "empty_preferred_fraction_total": frac_total,
        "empty_preferred_fraction_per_token": frac_per_token,
        "mean_margin_total": margin_total_sum / scored if scored else None,
        "mean_margin_per_token": margin_per_token_sum / scored if scored else None,
        "verdict": verdict,
        "per_record": per_record,
    }
    if cfg.score_policies:
        result["score_policies"] = [p.to_dict() for p in cfg.score_policies]
        result["policy_empty_preferred_fraction"] = {
            name: (count / scored if scored else None)
            for name, count in policy_empty_preferred.items()
        }
    return result


def _verdict(frac_total: float | None, frac_per_token: float | None) -> str:
    """Classify the emptiness cause from the two preference fractions."""
    if frac_total is None or frac_per_token is None:
        return "no_data"
    if frac_total >= 0.5 and frac_per_token < 0.5:
        return "length_bias_constraint_distortion"
    if frac_per_token >= 0.5:
        return "content_modeling_failure"
    return "populated_preferred"
