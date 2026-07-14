"""Canonical fixed-mask denoising NLL (candidate-invariant held-out loss).

For a held-out prompt/context ``c``, target program ``y``, fixed mask rate
``r``, and fixed mask seed ``s``::

    L_{r,s}(c, y) = -1/|M| * sum_{i in M} log p(y_i | y_without_M, c)

where ``M`` is derived from ``sha256(suite_version | record_id | rate | seed)``
so it is stable across dataset order, training configuration, and process
restarts. This is a *conditional denoising NLL* (masked-token NLL), not an
exact full-sequence likelihood.

Invariance contract (never read from the model config): mask rates, mask
positions, loss weighting. MDLM weighting, LTR suffix fusion, fidelity aux,
visible corruption, and best-of-N must not affect this number for a fixed
checkpoint.

Raw vs legal-support decomposition: ``L_raw`` scores over the full output
vocabulary; ``L_legal`` renormalizes over the grammar-allowed token support at
each gold prefix (the support the constrained decoder would see). The
``constraint_rescue_gap = L_raw - L_legal`` measures how much correctness the
grammar harness carries versus the model itself.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import torch
import torch.nn.functional as F

from slm_training.dsl.schema import ExampleRecord

DEFAULT_MASK_RATES: tuple[float, ...] = (0.15, 0.30, 0.50, 0.70, 0.85)

# Positions eligible for masking given a record + its target token ids.
PositionFilter = Callable[[ExampleRecord, list[int]], Sequence[int]]


@dataclass(frozen=True)
class DenoisingNLLConfig:
    suite_version: str = "v1"
    mask_rates: tuple[float, ...] = DEFAULT_MASK_RATES
    mask_seed: int = 0
    batch_size: int = 8
    # Compute grammar legal-support NLL (L_legal) alongside raw NLL.
    compute_legal_support: bool = True
    # Cap legal-support positions per record per rate (None = all masked).
    max_legal_positions: int | None = None

    def key(self) -> dict[str, Any]:
        return {
            "suite_version": self.suite_version,
            "mask_rates": list(self.mask_rates),
            "mask_seed": self.mask_seed,
        }


def _mask_rng(
    record_id: str, rate: float, *, suite_version: str, mask_seed: int
) -> random.Random:
    payload = f"{suite_version}|{record_id}|{rate:.4f}|{mask_seed}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def fixed_mask_positions(
    record_id: str,
    rate: float,
    *,
    suite_version: str,
    mask_seed: int,
    eligible: Sequence[int],
) -> list[int]:
    """Deterministic mask position sample for one record + rate.

    Independent of dataset order, model config, and process RNG state.
    """
    ordered = sorted(set(int(p) for p in eligible))
    if not ordered:
        return []
    n = max(1, round(rate * len(ordered)))
    n = min(n, len(ordered))
    rng = _mask_rng(record_id, rate, suite_version=suite_version, mask_seed=mask_seed)
    return sorted(rng.sample(ordered, n))


def _target_ids(model: Any, record: ExampleRecord) -> list[int]:
    from slm_training.models.twotower import _truncate_with_eos

    ids = model._encode_openui(
        record.openui,
        placeholders=list(record.placeholders or []),
        cache_key=record.id,
    )
    return _truncate_with_eos(
        ids, int(model.config.max_target_len), model.tokenizer.eos_id
    )


def _context_text(model: Any, record: ExampleRecord) -> str:
    """Mirror training conditioning (this is a model property, not a decoder knob)."""
    contract = None
    if bool(getattr(model.config, "slot_contract_in_context", False)):
        contract = model._resolve_slot_contract(record.prompt, record, record.design_md)
    return model._format_one_context(
        record.prompt,
        record.design_md,
        query_prompt=record.prompt,
        slot_contract=contract,
    )


def default_eligible_positions(model: Any, ids: list[int]) -> list[int]:
    """All predictable positions: everything except BOS (position 0) and pads."""
    pad_id = model.tokenizer.pad_id
    return [i for i in range(1, len(ids)) if ids[i] != pad_id]


def _legal_engine() -> Any | None:
    try:
        from slm_training.dsl.grammar.fastpath import engine_for_dsl
        from slm_training.models.grammar import active_dsl

        return engine_for_dsl(active_dsl())
    except Exception:  # noqa: BLE001
        return None


def _legal_support_ids(
    engine: Any,
    model: Any,
    record: ExampleRecord,
    ids: list[int],
    position: int,
) -> set[int] | None:
    """Grammar-allowed token ids at ``position`` given the *gold* prefix.

    Returns ``None`` when the support is broad/unknown (treated as the full
    vocabulary, i.e. unconstrained) — mirroring the constrained decoder's
    fallback behavior.
    """
    from slm_training.dsl.grammar.fastpath.token_map import allowed_id_set

    eos_id = model.tokenizer.eos_id
    if ids[position] == eos_id:
        # The grammar has no EOS terminal; termination is a model decision.
        return None
    prefix_text = model._decode_openui(
        ids[:position],
        placeholders=list(record.placeholders or []),
        cache_key=record.id,
    )
    try:
        synced = bool(engine.set_prefix(prefix_text))
    except Exception:  # noqa: BLE001
        return None
    if not synced:
        return None
    try:
        return allowed_id_set(model.tokenizer, engine.next_terminals())
    except Exception:  # noqa: BLE001
        return None


@dataclass
class _RateAccumulator:
    nll_sum: float = 0.0
    masked_tokens: int = 0
    legal_nll_sum: float = 0.0
    legal_positions: int = 0
    constrained_positions: int = 0
    gold_outside_support: int = 0
    # Per-record extrapolation inputs for the char-normalized metric.
    record_nll_means: list[tuple[float, int, int]] = field(default_factory=list)


@torch.no_grad()
def evaluate_denoising_nll(
    model: Any,
    records: list[ExampleRecord],
    *,
    config: DenoisingNLLConfig | None = None,
    position_filter: PositionFilter | None = None,
) -> dict[str, Any]:
    """Deterministic held-out denoising NLL for a TwoTower-style model.

    ``position_filter`` restricts maskable positions (e.g. binding-only or
    structural-only categories); masks are then sampled deterministically from
    the restricted set at each configured rate.
    """
    cfg = config or DenoisingNLLConfig()
    was_training = bool(getattr(model, "training", False))
    model.eval()
    tokenizer = model.tokenizer
    device = model.device_name

    engine = _legal_engine() if cfg.compute_legal_support else None
    legal_available = engine is not None

    accs: dict[float, _RateAccumulator] = {r: _RateAccumulator() for r in cfg.mask_rates}
    skipped: list[dict[str, str]] = []
    scored_records = 0

    batch_size = max(1, int(cfg.batch_size))
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        prepared: list[tuple[ExampleRecord, list[int], list[int]]] = []
        prompts: list[str] = []
        for record in batch:
            ids = _target_ids(model, record)
            eligible = default_eligible_positions(model, ids)
            if position_filter is not None:
                allowed = set(int(p) for p in position_filter(record, ids))
                eligible = [p for p in eligible if p in allowed]
            if not eligible:
                skipped.append({"id": record.id, "reason": "no_eligible_positions"})
                continue
            prepared.append((record, ids, eligible))
            prompts.append(_context_text(model, record))
        if not prepared:
            continue
        scored_records += len(prepared)

        ctx, ctx_pad = model._encode_context(prompts, cache_keys=None)
        from slm_training.models.twotower import _pad_batch

        target_ids = _pad_batch(
            [ids for _, ids, _ in prepared], tokenizer.pad_id, device=device
        )

        for rate in cfg.mask_rates:
            acc = accs[rate]
            noisy = target_ids.clone()
            mask = torch.zeros_like(target_ids, dtype=torch.bool)
            record_positions: list[list[int]] = []
            for row, (record, ids, eligible) in enumerate(prepared):
                positions = fixed_mask_positions(
                    record.id,
                    rate,
                    suite_version=cfg.suite_version,
                    mask_seed=cfg.mask_seed,
                    eligible=eligible,
                )
                record_positions.append(positions)
                for pos in positions:
                    noisy[row, pos] = tokenizer.mask_id
                    mask[row, pos] = True
            logits = model.denoiser(
                noisy, ctx, pad_id=tokenizer.pad_id, ctx_pad_mask=ctx_pad
            )
            log_probs = F.log_softmax(logits.float(), dim=-1)
            token_lp = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

            for row, (record, ids, _eligible) in enumerate(prepared):
                positions = record_positions[row]
                if not positions:
                    continue
                row_nll = 0.0
                for pos in positions:
                    row_nll += -float(token_lp[row, pos].item())
                acc.nll_sum += row_nll
                acc.masked_tokens += len(positions)
                acc.record_nll_means.append(
                    (
                        row_nll / len(positions),
                        len(ids),
                        max(1, len(record.openui)),
                    )
                )

                if engine is None:
                    continue
                legal_positions = positions
                if (
                    cfg.max_legal_positions is not None
                    and len(legal_positions) > cfg.max_legal_positions
                ):
                    legal_positions = legal_positions[: cfg.max_legal_positions]
                for pos in legal_positions:
                    support = _legal_support_ids(engine, model, record, ids, pos)
                    gold_id = int(ids[pos])
                    if support is None:
                        # Unconstrained position — legal NLL equals raw NLL.
                        acc.legal_nll_sum += -float(token_lp[row, pos].item())
                        acc.legal_positions += 1
                        continue
                    if gold_id not in support:
                        support = set(support)
                        support.add(gold_id)
                        acc.gold_outside_support += 1
                    idx = torch.tensor(
                        sorted(support), dtype=torch.long, device=logits.device
                    )
                    row_logits = logits[row, pos].float()
                    support_lse = torch.logsumexp(row_logits[idx], dim=0)
                    legal_lp = float(row_logits[gold_id].item()) - float(
                        support_lse.item()
                    )
                    acc.legal_nll_sum += -legal_lp
                    acc.legal_positions += 1
                    acc.constrained_positions += 1

    rates_report: dict[str, Any] = {}
    total_nll = 0.0
    total_tokens = 0
    total_legal_nll = 0.0
    total_legal_positions = 0
    bits_num = 0.0
    bits_den = 0.0
    for rate in cfg.mask_rates:
        acc = accs[rate]
        mean_nll = acc.nll_sum / acc.masked_tokens if acc.masked_tokens else None
        legal_mean = (
            acc.legal_nll_sum / acc.legal_positions if acc.legal_positions else None
        )
        gap = (
            (acc.nll_sum / acc.masked_tokens) - legal_mean
            if acc.masked_tokens and legal_mean is not None
            else None
        )
        rates_report[f"{rate:.2f}"] = {
            "mean_nll": mean_nll,
            "masked_tokens": acc.masked_tokens,
            "legal_mean_nll": legal_mean if legal_available else None,
            "constraint_rescue_gap": gap if legal_available else None,
            "constrained_fraction": (
                acc.constrained_positions / acc.legal_positions
                if acc.legal_positions
                else None
            ),
            "gold_outside_support": acc.gold_outside_support,
        }
        total_nll += acc.nll_sum
        total_tokens += acc.masked_tokens
        total_legal_nll += acc.legal_nll_sum
        total_legal_positions += acc.legal_positions
        for mean, n_tokens, n_chars in acc.record_nll_means:
            # Extrapolate masked-position NLL to the full target, normalized by
            # canonical target bytes — approximate but tokenizer-invariant.
            bits_num += mean * n_tokens
            bits_den += n_chars

    mean_nll_all = total_nll / total_tokens if total_tokens else None
    legal_mean_all = (
        total_legal_nll / total_legal_positions if total_legal_positions else None
    )
    if was_training:
        model.train()
    return {
        **cfg.key(),
        "n_records": scored_records,
        "n_skipped": len(skipped),
        "skipped": skipped[:20],
        "rates": rates_report,
        "aggregate": {
            "mean_nll": mean_nll_all,
            "masked_tokens": total_tokens,
            "legal_mean_nll": legal_mean_all if legal_available else None,
            "constraint_rescue_gap": (
                mean_nll_all - legal_mean_all
                if legal_available
                and mean_nll_all is not None
                and legal_mean_all is not None
                else None
            ),
        },
        # Approximate bits-per-byte over len(mask_rates) extrapolated passes.
        "bits_per_char": (
            bits_num / (math.log(2.0) * bits_den) if bits_den else None
        ),
        "nll_per_char": (bits_num / bits_den if bits_den else None),
        "legal_support_available": legal_available,
    }
