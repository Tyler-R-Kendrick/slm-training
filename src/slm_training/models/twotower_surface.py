"""Surface-level text handling and decode bookkeeping.

One responsibility: canonicalising and repairing emitted OpenUI text, pooling
context, and folding the per-state engine statistics a decode accumulates.

Extracted from ``TwoTowerModel``. See ``docs/design/code-quality-contract.md``.
"""

from __future__ import annotations

import re

import torch

from slm_training.models.decode_stats import (
    get_active_stats,
)


_QUOTED_SPAN_RE = re.compile(r'("(?:\\.|[^"\\])*")')
_REPEATED_EQUALS_RE = re.compile(r"\s*=\s*=+\s*")
_DANGLING_EQUALS_RE = re.compile(r",\s*=\s*(?=[)\]])")


def note_admit_rejection(run: int) -> int:
    """N12 (read-only): record one admit-probe rejection.

    ``run`` is the caller's count of consecutive rejections with no
    intervening commit; the returned value is the extended run. Nothing in
    the decode path reads the counters back, so the proposal sequence and
    the emitted canvas are identical whether or not a collector is active.
    """
    run = int(run) + 1
    stats = get_active_stats()
    if stats is not None:
        stats.admit_probe_rejections += 1
        if run > stats.admit_probe_reject_run_max:
            stats.admit_probe_reject_run_max = run
    return run


def fold_state_engine_stats(states: list | None) -> None:
    """Fold each state engine's lifetime counters into the active stats.

    Called once when a decode path's request-local states go out of
    scope; engines are per-state, so a single fold never double-counts.
    """
    if not states:
        return
    from slm_training.models.decode_stats import collect_engine_stats

    for state in states:
        engine = getattr(state, "engine", None)
        if engine is not None:
            collect_engine_stats(engine)


def pool_context(context: torch.Tensor, pad_mask: torch.Tensor | None) -> torch.Tensor:
    if pad_mask is None:
        return context.mean(dim=1)
    visible = (~pad_mask).unsqueeze(-1).to(context.dtype)
    return (context * visible).sum(dim=1) / visible.sum(dim=1).clamp(min=1.0)


def repair_surface_syntax(text: str) -> str:
    """Repair local token-boundary artifacts without inventing layout content."""
    parts = _QUOTED_SPAN_RE.split(text)
    for index in range(0, len(parts), 2):
        parts[index] = _REPEATED_EQUALS_RE.sub(" = ", parts[index])
        parts[index] = _DANGLING_EQUALS_RE.sub("", parts[index])
    return "".join(parts)


def canonical_valid_openui(text: str) -> str | None:
    """Return serialized OpenUI if parseable and non-trivial; else None."""
    try:
        from slm_training.dsl.parser import validate
    except TimeoutError:
        raise
    except Exception:  # noqa: BLE001
        return None
    try:
        program = validate(text)
    except TimeoutError:
        raise
    except Exception:  # noqa: BLE001
        return None
    ser = (program.serialized or text).strip()
    compact = ser.replace(" ", "")
    if "Stack([])" in compact or "Stack([]," in compact:
        return None
    if "Card([])" in compact:
        return None
    if "root=" not in compact and "root =" not in ser:
        return None
    return ser
