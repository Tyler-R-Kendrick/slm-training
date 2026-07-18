"""Deterministic strata balancing for the LDI1-02 causal preference corpus.

Balances admitted ``(DecisionStateV2, ObjectiveView)`` items across configurable
strata *without duplicating evidence*: each stratum is capped by sampling without
replacement, so a thin stratum can never be inflated to fabricate support. The
sampling is deterministic in ``(items, strata, seed)`` and reports the before /
after distribution, exclusions, and effective sample counts.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from slm_training.harnesses.preference.decision_events_v2 import (
    DecisionStateV2,
    ObjectiveView,
)

__all__ = ["CausalTrainingItem", "BalanceStratum", "balance_items"]

BalanceStratum = Literal[
    "source_suite",
    "decision_kind",
    "abstract_state_role",
    "prompt_group",
    "objective_signature",
    "split",
]
_VALID_STRATA: tuple[str, ...] = (
    "source_suite",
    "decision_kind",
    "abstract_state_role",
    "prompt_group",
    "objective_signature",
    "split",
)


@dataclass(frozen=True)
class CausalTrainingItem:
    """One admitted training decision: an exact state plus its objective view."""

    state: DecisionStateV2
    view: ObjectiveView
    source_suite: str = "default"


def _stratum_value(item: CausalTrainingItem, stratum: str) -> str:
    if stratum == "source_suite":
        return item.source_suite
    if stratum == "decision_kind":
        return item.state.decision_kind
    if stratum == "abstract_state_role":
        return item.state.abstract_state_role
    if stratum == "prompt_group":
        return item.state.group_id
    if stratum == "objective_signature":
        return item.view.materializer_id
    if stratum == "split":
        return str(item.state.split)
    raise ValueError(f"unknown balance stratum {stratum!r}")


def _stratum_key(item: CausalTrainingItem, strata: Sequence[str]) -> tuple[str, ...]:
    return tuple(_stratum_value(item, stratum) for stratum in strata)


def balance_items(
    items: Sequence[CausalTrainingItem],
    *,
    strata: Sequence[BalanceStratum],
    seed: int,
    per_stratum: int | None = None,
    drop_nontrainable: bool = True,
) -> tuple[list[CausalTrainingItem], dict[str, Any]]:
    """Return balanced items and a before/after report.

    Non-trainable (constraint-shadow) views are excluded up front — legality is
    not a preference label. Each stratum is then capped at ``per_stratum`` (or, by
    default, the smallest non-empty stratum size, equalizing support) by sampling
    without replacement under a seeded RNG, so no evidence is duplicated.
    """
    if not strata:
        raise ValueError("at least one stratum is required")
    # Validate strata at the trust boundary, before inspecting items, so an
    # unknown stratum is rejected even when the input is empty or fully excluded.
    for stratum in strata:
        if stratum not in _VALID_STRATA:
            raise ValueError(f"unknown balance stratum {stratum!r}")

    kept: list[CausalTrainingItem] = []
    seen_state_ids: set[str] = set()
    excluded_nontrainable = 0
    for item in items:
        if drop_nontrainable and not item.view.trainable:
            excluded_nontrainable += 1
            continue
        # The no-duplication contract must reject pre-existing duplicate evidence,
        # not only avoid duplicating during sampling.
        if item.state.state_id in seen_state_ids:
            raise ValueError(f"duplicate training state {item.state.state_id!r}")
        seen_state_ids.add(item.state.state_id)
        kept.append(item)

    grouped: dict[tuple[str, ...], list[CausalTrainingItem]] = {}
    for item in kept:
        grouped.setdefault(_stratum_key(item, strata), []).append(item)

    before = {"__".join(key): len(group) for key, group in grouped.items()}
    if not grouped:
        return [], {
            "strata": list(strata),
            "seed": seed,
            "cap": 0,
            "before": {},
            "after": {},
            "excluded_nontrainable": excluded_nontrainable,
            "excluded_by_cap": 0,
            "effective_count": 0,
        }

    cap = per_stratum if per_stratum is not None else min(len(g) for g in grouped.values())
    if cap < 0:
        raise ValueError("per_stratum must be non-negative")

    rng = random.Random(seed)
    balanced: list[CausalTrainingItem] = []
    after: dict[str, int] = {}
    excluded_by_cap = 0
    for key in sorted(grouped):
        # Deterministic order before the seeded shuffle so results are replayable.
        group = sorted(grouped[key], key=lambda it: it.state.state_id)
        take = min(cap, len(group))
        if take < len(group):
            group = rng.sample(group, take)  # without replacement -> no duplication
            group = sorted(group, key=lambda it: it.state.state_id)
        excluded_by_cap += len(grouped[key]) - take
        balanced.extend(group)
        after["__".join(key)] = take

    report = {
        "strata": list(strata),
        "seed": seed,
        "cap": cap,
        "before": before,
        "after": after,
        "excluded_nontrainable": excluded_nontrainable,
        "excluded_by_cap": excluded_by_cap,
        "effective_count": len(balanced),
    }
    return balanced, report
